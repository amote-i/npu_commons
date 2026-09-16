import os
import time

import pytest
import sgl_kernel_npu  # noqa: F401  registers npu ops before pytestmark
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  makes torch.ops.npu namespace available
from sgl_kernel_npu.fla.solve_tril import solve_tril_npu as solve_tril
from utils import require_npu_op

NPU_DEVICE = "npu"

# ---------------------------------------------------------------------------
# Incident reproduction: merge_16x16_to_64x64_inverse_kernel aicore timeout
# (2026-09-14, rtStreamSynchronize 507014, blockDim=35424, fault blk=32787).
#
# Reproduces the exact launch: varlen batch of 24 packed sequences totalling
# 47232 tokens -> NT=738, H=48, grid = NT * H = 35424. Tweak _INCIDENT_SEQS /
# _INCIDENT_H if shape logging in serving reveals the real packing.
#
# Run on a dedicated card (a reproduced hang wedges the process until the
# watchdog fires, and possibly longer):
#   ASCEND_RT_VISIBLE_DEVICES=<n> python test_solve_tril.py
#
# Reading the result:
#   - RuntimeError aicore timeout / 507014  -> reproduced. Compare the new
#     PLOG against the incident one: ProcLogicCqReport errType=0x4,
#     PrintCoreInfo's blk field (incident had blk=32787, just past 32768),
#     GetArgsInfo blockDim=35424.
#   - clean exit                          -> not reproducible standalone;
#     the printed timing/correctness of both launches is then the takeaway.
# ---------------------------------------------------------------------------

_INCIDENT_SEQS = [2048] * 23 + [128]  # 47232 tokens, NT=738
_INCIDENT_H = 48

LEGACY_SKIP_REASON = "solve_tril API is not consistent with upstream fla."


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    else:
        assert error_rate < ratio, msg


def print_diff(name, ref, tri, atol=0.005):
    abs_diff = torch.abs(ref - tri)
    max_abs_diff = abs_diff.max().item()
    print(f"[{name}] Max absolute difference: {max_abs_diff:.6f}")
    if max_abs_diff > atol:
        print(f"Exceeds tolerance ({atol})!")


@pytest.mark.skip(reason=LEGACY_SKIP_REASON)
@require_npu_op("triangular_inverse")
@pytest.mark.parametrize(
    ("B", "T", "H", "chunk_size"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-chunk_size{}".format(*test))
        for test in [
            (1, 63, 1, 16),
            (2, 500, 4, 32),
            (2, 1000, 5, 64),
            (3, 1024, 6, 64),
            (4, 2048, 8, 64),
        ]
    ],
)
def test_solve_tril(B, T, H, chunk_size):
    # do not randomly initialize A otherwise the inverse is not stable
    k = F.normalize(
        torch.randn((B, H, T, 64), dtype=torch.float32, device=NPU_DEVICE), dim=-1
    )
    torch.npu.synchronize()
    # Pad the second-to-last dimension (T) to be a multiple of chunk_size
    padding_size = (chunk_size - T % chunk_size) % chunk_size
    k_padded = F.pad(k, (0, 0, 0, padding_size, 0, 0, 0, 0))
    torch.npu.synchronize()
    k_padded = k_padded.reshape(B, H, -1, chunk_size, 64)
    torch.npu.synchronize()
    A = (k_padded @ k_padded.transpose(-1, -2)).tril(-1).npu()
    torch.npu.synchronize()

    ref = torch.inverse(
        A
        + torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)[None, None, None, ...]
    )
    torch.npu.synchronize()
    ref = ref.reshape(B, H, -1, chunk_size)[:, :, :T, :]

    torch.npu.synchronize()
    tri = solve_tril(
        A.reshape(B, H, -1, chunk_size)[:, :, :T, :].transpose(1, 2)
    ).transpose(1, 2)
    torch.npu.synchronize()

    assert_close("solve_tril", ref, tri, 0.0001)


def _build_case(seqs, H, seed=0):
    """Varlen-packed banded input [1, T, H, 64] for solve_tril_npu, in the
    same layout chunk_scaled_dot_kkt feeds in production: per 64-token chunk,
    the chunk's rows hold (k @ k^T).tril(-1) of that sequence."""
    total = int(sum(seqs))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    k = F.normalize(
        torch.randn((1, H, total, 64), dtype=torch.float32, generator=gen), dim=-1
    ).to(NPU_DEVICE)

    A = torch.zeros(1, total, H, 64, dtype=torch.float32, device=NPU_DEVICE)
    nt, bos = 0, 0
    for length in seqs:
        nc = -(-length // 64)
        nt += nc
        ks = F.pad(k[0, :, bos : bos + length, :], (0, 0, 0, nc * 64 - length))
        blk = (ks.reshape(H, nc, 64, 64) @ ks.reshape(H, nc, 64, 64).transpose(-1, -2))
        A[0, bos : bos + nc * 64] = blk.tril(-1).permute(1, 2, 0, 3).reshape(
            nc * 64, H, 64
        )
        bos += length

    cu_list = [0]
    for length in seqs:
        cu_list.append(cu_list[-1] + length)
    cu = torch.tensor(cu_list, dtype=torch.long).to(NPU_DEVICE)
    return A, cu, nt


def _diff_vs_reference(Ai, A, seqs):
    """Max abs diff vs (I - A)^-1 (the kernel's convention per its source);
    also reports (I + A)^-1 so a sign surprise is self-diagnosing."""
    eye = torch.eye(64, dtype=torch.float32, device=A.device)
    out_d = {"minus": 0.0, "plus": 0.0}
    bos = 0
    for length in seqs:
        nc = -(-length // 64)
        blk = A[0, bos : bos + nc * 64].reshape(nc, 64, *A.shape[2:]).permute(
            2, 0, 1, 3
        )
        out = Ai[0, bos : bos + nc * 64].reshape(nc, 64, *Ai.shape[2:]).permute(
            2, 0, 1, 3
        )
        m = (torch.arange(nc * 64, device=A.device) < length).reshape(1, nc, 64, 1)
        for tag, ref in (
            ("minus", torch.inverse(eye - blk)),
            ("plus", torch.inverse(eye + blk)),
        ):
            out_d[tag] = max(out_d[tag], ((out - ref).abs() * m).max().item())
        bos += length
    return out_d


def run_incident_repro():
    # Small same-H control first: proves the card/kernel are healthy (and
    # warms the H=48 varlen JIT specialization) before the incident launch.
    control_seqs = [2048]
    A, cu, nt = _build_case(control_seqs, _INCIDENT_H)
    t0 = time.perf_counter()
    Ai = solve_tril(A=A, cu_seqlens=cu, output_dtype=torch.float32)
    torch.npu.synchronize()
    d = _diff_vs_reference(Ai, A, control_seqs)
    print(
        f"[control ] grid={nt * _INCIDENT_H} time={(time.perf_counter() - t0) * 1e3:.1f}ms "
        f"diff_(I-A)^-1={d['minus']:.3e} diff_(I+A)^-1={d['plus']:.3e}",
        flush=True,
    )
    del A, Ai

    A, cu, nt = _build_case(_INCIDENT_SEQS, _INCIDENT_H)
    grid = nt * _INCIDENT_H
    print(
        f"[incident] tokens={sum(_INCIDENT_SEQS)} seqs={len(_INCIDENT_SEQS)} "
        f"NT={nt} H={_INCIDENT_H} grid={grid}",
        flush=True,
    )
    t0 = time.perf_counter()
    Ai = solve_tril(A=A, cu_seqlens=cu, output_dtype=torch.float32)
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) * 1e3
    d = _diff_vs_reference(Ai, A, _INCIDENT_SEQS)
    print(
        f"[incident] completed time={dt:.1f}ms "
        f"diff_(I-A)^-1={d['minus']:.3e} diff_(I+A)^-1={d['plus']:.3e}",
        flush=True,
    )
    assert d["minus"] < 5e-3, f"completed but wrong result (diff={d['minus']:.3e})"


@pytest.mark.skipif(
    os.getenv("SGL_REPRO") != "1",
    reason="destructive incident reproduction; set SGL_REPRO=1 on a dedicated card",
)
def test_solve_tril_incident_repro():
    run_incident_repro()


if __name__ == "__main__":
    run_incident_repro()
