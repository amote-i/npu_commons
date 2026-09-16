# Adapted and Merge from
#   https://github.com/sglang/python/sglang/srt/layers/attention/fla/wy_fast.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, By Triton_Ascend & sglang_ascend

import os
import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sgl_kernel_npu.fla.utils import (
    exp,
    prepare_chunk_indices,
    prepare_chunk_offsets,
    safe_exp,
)


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel_npu_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    T_max = T
    i_t_o, _ = tl.program_id(0), tl.program_id(1)
    for i_bh in range(H):
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n, i_t = tl.load(chunk_indices + i_t_o * 2).to(tl.int32), tl.load(
                chunk_indices + i_t_o * 2 + 1
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
                cu_seqlens + i_n + 1
            ).to(tl.int32)
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T

        offs_t = tl.arange(0, BT)
        global_offs_t = i_t * BT + offs_t
        mask_t = global_offs_t < T

        offs_t_2d = global_offs_t[:, None]
        offs_bt = tl.arange(0, BT)[None, :]
        ptr_A = A + (bos * H + i_h) * BT + offs_t_2d * (H * BT) + offs_bt * 1
        mask_A = mask_t[:, None]
        b_A = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

        ptr_g = g + bos + i_h * T_max + global_offs_t
        b_g = tl.exp(tl.load(ptr_g, mask=mask_t, other=0.0)).to(tl.float32)

        ptr_beta = beta + bos + i_h * T_max + global_offs_t
        b_beta = tl.load(ptr_beta, mask=mask_t, other=0.0).to(tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            # --- load v (BTxBV) ---
            offs_v = i_v * BV + tl.arange(0, BV)[None, :]
            mask_v = (mask_t[:, None]) & (offs_v < V)
            # orig strides (H * V, 1)
            ptr_v = v + (bos * H + i_h) * V + offs_t_2d * (H * V) + offs_v * 1
            b_v = tl.load(ptr_v, mask=mask_v, other=0.0).to(tl.float32)

            b_vb = b_v * b_beta[:, None]
            b_u = tl.dot(b_A, b_vb, allow_tf32=False)
            ptr_u = u + (bos * H + i_h) * V + offs_t_2d * (H * V) + offs_v * 1
            tl.store(ptr_u, b_u.to(ptr_u.dtype.element_ty), mask=mask_v)

        for i_k in range(tl.cdiv(K, BK)):
            offs_k = i_k * BK + tl.arange(0, BK)[None, :]
            mask_k = (mask_t[:, None]) & (offs_k < K)
            # orig strides (Hg * K, 1)
            ptr_k = (
                k
                + (bos * Hg + i_h // (H // Hg)) * K
                + offs_t_2d * (Hg * K)
                + offs_k * 1
            )
            b_k = tl.load(ptr_k, mask=mask_k, other=0.0).to(tl.float32)

            b_kb = b_k * b_beta[:, None] * b_g[:, None]
            b_w = tl.dot(b_A, b_kb)
            ptr_w = w + (bos * H + i_h) * K + offs_t_2d * (H * K) + offs_k * 1
            tl.store(ptr_w, b_w.to(ptr_w.dtype.element_ty), mask=mask_k)


# Debug helper: set SGLK_WY_FAST_DEBUG=1 to log every recompute_w_u launch
# (shapes, grid, varlen boundaries, elapsed time). Run together with
# ASCEND_LAUNCH_BLOCKING=1 so a hung kernel surfaces synchronously: the last
# "launching" line without a following "launched OK" is the culprit.
_DEBUG_WY_FAST = os.environ.get("SGLK_WY_FAST_DEBUG", "0") == "1"


def _debug_log_launch(
    tag,
    *,
    NT,
    B,
    T,
    H,
    Hg,
    K,
    V,
    BT,
    BK,
    BV,
    k,
    v,
    A,
    beta,
    g_cumsum,
    cu_seqlens,
    chunk_indices,
):
    cu = cu_seqlens.tolist() if cu_seqlens is not None else None
    stats = []
    for name, t in (
        ("k", k),
        ("v", v),
        ("beta", beta),
        ("g", g_cumsum),
        ("A", A),
    ):
        # NOTE: forces a device sync per tensor; debug only. Signed min/max on
        # purpose: the kernel applies tl.exp(g) directly, so only a *positive*
        # g above ~88 overflows to inf (a large negative g just underflows).
        nan = int(torch.isnan(t).sum().item())
        inf = int(torch.isinf(t).sum().item())
        tmin = float(t.min().item())
        tmax = float(t.max().item())
        stats.append(f"{name} nan={nan} inf={inf} min={tmin:.3g} max={tmax:.3g}")
    ci = None
    if chunk_indices is not None:
        ci_flat = chunk_indices.flatten().tolist()
        ci = (
            f"n_chunks={len(ci_flat) // 2} "
            f"first={ci_flat[:8]} last={ci_flat[-8:]}"
        )
    print(
        f"[wy_fast] pid={os.getpid()} recompute_w_u {tag}: "
        f"grid=({NT},{B}) T={T} H={H} Hg={Hg} "
        f"K={K} V={V} BT={BT} BK={BK} BV={BV} "
        f"varlen={cu_seqlens is not None} cu_seqlens={cu} chunk_indices={ci} "
        f"k={tuple(k.shape)}x{tuple(k.stride())} "
        f"v={tuple(v.shape)}x{tuple(v.stride())} "
        f"A={tuple(A.shape)}x{tuple(A.stride())} "
        f"beta={tuple(beta.shape)}x{tuple(beta.stride())} "
        f"g={tuple(g_cumsum.shape)}x{tuple(g_cumsum.stride())} "
        f"dev={k.device} warps=4 stages=3 | inputs: {'; '.join(stats)}",
        flush=True,
    )


def recompute_w_u_fwd_npu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = 128
    BV = 128
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    # Pre-transpose tensors outside the kernel to ensure contiguous memory access within the kernel
    # avoiding scattered (non-contiguous) access that may lead to axis expansion
    beta = beta.transpose(1, 2).contiguous()
    g_cumsum = g_cumsum.transpose(1, 2).contiguous()
    if _DEBUG_WY_FAST:
        _debug_log_launch(
            "launching",
            NT=NT,
            B=B,
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            k=k,
            v=v,
            A=A,
            beta=beta,
            g_cumsum=g_cumsum,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        t0 = time.perf_counter()
    recompute_w_u_fwd_kernel_npu_kernel[(NT, B)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=3,
    )
    if _DEBUG_WY_FAST:
        # Elapsed is only meaningful with ASCEND_LAUNCH_BLOCKING=1 (sync launch);
        # without it the launch returns immediately before the kernel finishes.
        print(
            f"[wy_fast] pid={os.getpid()} dev={k.device} recompute_w_u "
            f"launched OK in {(time.perf_counter() - t0) * 1e3:.1f} ms",
            flush=True,
        )
    return w, u
