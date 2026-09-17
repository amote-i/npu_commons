#!/usr/bin/env python3
"""Offline reproduction harness for the solve_tril aicore-timeout hang.

Target fault (from plog):
    errType=0x4 (task timeout) / retCode=0x25 [aicore timeout]
    fault kernel_name = merge_16x16_to_64x64_inverse_kernel_0
    grid total blocks (blockDim) = 35424, MTE error info non-zero

The hang is sporadic under serving, so this harness fuzzes the parameter
space that plausibly triggers it, with these properties:

  * every trial runs in its own subprocess (own session) with a wall-clock
    timeout, so a wedged trial is SIGKILLed and the harness survives;
  * the manifest is written BEFORE a trial starts, so even a hard wedge
    leaves the exact failing config on disk;
  * a RuntimeError matching the aicore-timeout signature (507014 /
    "aicore timeout") is classified as AICORE_TIMEOUT and stops the run:
    that IS the reproduction we are hunting;
  * numeric output is checked against a torch reference to also catch
    silent corruption, not just hangs.

Scenario axes (see scenarios() below):
  - varlen / non-varlen, ragged cu_seqlens (zero-length seq, 1-token seq,
    lengths at 64 and 1216 boundaries: LARGE_BLOCK_T = 608*2 for stage 1);
  - grids whose total program count equals exactly 35424 and neighbours,
    reverse-engineered from the plog blockDim field;
  - non-contiguous A views (last-dim slice, permuted view -- the same
    pattern the skipped pytest passes) -- hang candidates, no numeric check;
  - num_warps x num_stages sweep on the merge kernel launch;
  - plain vs reorder_all_masked merge variant cross-check;
  - dtype / magnitude variants; --repeat for launch-count soak;
  - optional --bg-load N background matmul workers to mimic serving
    device pressure; --launch-blocking for error-attribution runs.

Examples:
    python scripts/repro_solve_tril_hang.py --device 0
    python scripts/repro_solve_tril_hang.py --list
    python scripts/repro_solve_tril_hang.py --shuffle --seed 7 --soak-min 120
    python scripts/repro_solve_tril_hang.py --only 35424 --repeat 50
    python scripts/repro_solve_tril_hang.py --launch-blocking --only varlen
    # replay a recorded hit:
    python scripts/repro_solve_tril_hang.py --child --device 0 --config '<json>'
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "python"
if SRC_ROOT.is_dir():
    sys.path.insert(0, str(SRC_ROOT))

# mirrors solve_tril.py internals
BT = 64
LARGE_BLOCK_T = 608 * 2

EXIT_OK = 0
EXIT_MISMATCH = 10
EXIT_AICORE_TIMEOUT = 20
EXIT_ERROR = 30

TIMEOUT_MARKERS = (
    "507014",
    "0x7150025",
    "aicore timeout",
    "task timeout",
    "kernel task happen error",
)

# numeric reference is skipped above this many 64x64 blocks (memory)
CHECK_BLOCK_LIMIT = 50_000


def sc(
    name,
    *,
    B=1,
    T=64,
    lens=None,
    H=8,
    out_dtype="bf16",
    a_dtype="fp32",
    scale=0.1,
    contig="c",
    variant="plain",
    cpp=1,
    num_warps=4,
    num_stages=3,
    repeat=1,
):
    return {
        "name": name,
        "B": B,
        "T": T,
        "lens": lens,
        "H": H,
        "out_dtype": out_dtype,
        "a_dtype": a_dtype,
        "scale": scale,
        "contig": contig,
        "variant": variant,
        "cpp": cpp,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "repeat": repeat,
    }


def scenarios():
    S = []

    # ---- production-like baselines (contiguous fp32 A, varlen, bf16 out) ----
    S.append(sc("prod_varlen_ragged", lens=[137, 64, 63, 1, 65, 128, 200], H=16))
    S.append(sc("prod_varlen_single_long", lens=[4096], H=32))
    S.append(sc("prod_varlen_many_short", lens=[7] * 64, H=8))
    S.append(sc("prod_varlen_zero_len_seq", lens=[0, 500, 0, 300], H=8))
    S.append(sc("prod_varlen_lbt_boundary", lens=[1216, 1215, 1217, 2432], H=8))
    S.append(sc("prod_varlen_bt_boundary", lens=[64 * k + d for k, d in [(3, 0), (5, 1), (2, 63), (1, 65)]], H=16))

    # ---- non-varlen path ----
    S.append(sc("nonvarlen_t64", B=2, T=64, H=8))
    S.append(sc("nonvarlen_t65", B=2, T=65, H=8))
    S.append(sc("nonvarlen_t63", B=1, T=63, H=8))
    S.append(sc("nonvarlen_t1", B=1, T=1, H=8))
    S.append(sc("nonvarlen_lbt_boundary", B=1, T=1216, H=8))
    S.append(sc("nonvarlen_lbt_boundary_pm1", B=1, T=1217, H=8, out_dtype="fp32"))
    S.append(sc("nonvarlen_t2432", B=1, T=2432, H=8))

    # ---- exact plog grid: total programs (NT * B * H) == 35424 ----
    S.append(sc("grid35424_3x32x123chunks", lens=[7872] * 3, H=32))            # NT=369, BH=96
    S.append(sc("grid35424_1seq_h32", lens=[70848], H=32))                     # NT=1107, BH=32
    S.append(sc("grid35424_b9_h32", B=9, T=7872, H=32))                        # NT=123, BH=288
    S.append(sc("grid_2x_70848", lens=[70848], H=64))                          # 2x program count

    # ---- non-contiguous A (hang candidates; numeric check auto-skipped) ----
    S.append(sc("noncontig_lastdim_view", lens=[512, 256], H=8, contig="lastdim"))
    S.append(sc("noncontig_permute_view", lens=[512, 256], H=8, contig="permute"))
    S.append(sc("noncontig_permute_nonvarlen", B=2, T=1024, H=8, contig="permute", out_dtype="fp32"))

    # ---- dtype / magnitude ----
    S.append(sc("out_fp32", lens=[1024, 512], H=16, out_dtype="fp32"))
    S.append(sc("out_fp16", lens=[1024, 512], H=16, out_dtype="fp16"))
    S.append(sc("a_bf16", lens=[1024, 512], H=16, a_dtype="bf16"))
    S.append(sc("scale_10", lens=[1024, 512], H=16, scale=10.0))
    S.append(sc("scale_100", lens=[1024, 512], H=16, scale=100.0))

    # ---- launch-config sweep on the faulting kernel ----
    for nw in (1, 2, 4, 8):
        for ns in (1, 2, 3, 4):
            if (nw, ns) == (4, 3):
                continue  # already the default covered by baselines
            S.append(sc(f"sweep_w{nw}_s{ns}", lens=[2048, 1024, 64], H=16, num_warps=nw, num_stages=ns))

    # ---- variant cross-check: reorder_all_masked ----
    S.append(sc("reorder_cpp1", lens=[2048, 1024], H=16, variant="reorder", cpp=1))
    S.append(sc("reorder_cpp2", lens=[2048, 1024], H=16, variant="reorder", cpp=2))
    S.append(sc("reorder_cpp4", lens=[2048, 1024], H=16, variant="reorder", cpp=4))

    # ---- launch-count soak candidate (cheap; amplify with --repeat) ----
    S.append(sc("soak_small", lens=[64, 65, 63], H=4, repeat=200))
    return S


# ---------------------------------------------------------------- parent ----


def is_hit(status):
    return status in ("HANG", "AICORE_TIMEOUT")


def kill_process_group(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()


def run_env_info(args):
    """Collect torch/torch_npu/triton versions via a short-lived child."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--env-info", "--device", str(args.device)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
        for line in out.splitlines():
            if line.startswith("RESULT:"):
                return json.loads(line[len("RESULT:"):])
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return {}


def spawn_bg_load(args):
    procs = []
    for _ in range(args.bg_load):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--bg-load-worker",
            "--device",
            str(args.device),
        ]
        procs.append(subprocess.Popen(cmd, start_new_session=True))
    return procs


def main_parent(args):
    all_sc = scenarios()
    if args.only:
        all_sc = [s for s in all_sc if args.only in s["name"]]
    if not all_sc:
        print(f"no scenario matches --only {args.only!r}; use --list to inspect")
        return 1

    manifest = open(args.manifest, "a", encoding="utf-8")

    def emit(rec):
        manifest.write(json.dumps(rec, default=str) + "\n")
        manifest.flush()

    env_info = run_env_info(args)
    emit({"type": "env", "ts": time.time(), "argv": sys.argv, "env": env_info})
    print(f"[harness] {len(all_sc)} scenarios, per-trial timeout {args.timeout}s")
    print(f"[harness] torch={env_info.get('torch')} torch_npu={env_info.get('torch_npu')} "
          f"triton={env_info.get('triton')} device={env_info.get('device_name')}")

    bg_procs = spawn_bg_load(args) if args.bg_load else []
    if bg_procs:
        print(f"[harness] {len(bg_procs)} background load workers running")

    order = list(range(len(all_sc)))
    if args.shuffle:
        random.Random(args.seed).shuffle(order)

    counts = {}
    hits = []
    deadline = time.time() + args.soak_min * 60 if args.soak_min else None
    trial_id = 0
    aborted = False
    try:
        while True:
            for idx in order:
                if deadline and time.time() > deadline:
                    aborted = True
                    break
                cfg = dict(all_sc[idx])
                cfg["repeat"] = cfg.get("repeat", 1) * args.repeat
                cfg["seed"] = args.seed * 100003 + trial_id
                trial_id += 1

                emit({"type": "start", "trial": trial_id, "config": cfg})
                rec = run_trial_subprocess(cfg, args)
                rec.update({"type": "result", "trial": trial_id})
                emit(rec)
                counts[rec["status"]] = counts.get(rec["status"], 0) + 1

                flag = "  <-- HIT" if is_hit(rec["status"]) else ""
                print(f"[trial {trial_id:>3}] {rec['status']:<14} {cfg['name']}{flag}", flush=True)

                if is_hit(rec["status"]):
                    hits.append(rec)
                    hit_path = Path(args.manifest).with_name("repro_solve_tril_hit.json")
                    hit_path.write_text(json.dumps(rec, indent=2, default=str))
                    replay = (f"{sys.executable} {Path(__file__).resolve()} --child "
                              f"--device {args.device} --config "
                              f"'{json.dumps(cfg)}'")
                    print(f"[harness] HIT saved to {hit_path}\n[harness] replay with:\n  {replay}")
                    if args.stop_on_hit:
                        aborted = True
                        break
                if args.max_trials and trial_id >= args.max_trials:
                    aborted = True
                    break
            if aborted or not args.soak_min:
                break
    finally:
        for p in bg_procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                p.kill()

    print("\n===== summary =====")
    print(f"trials: {trial_id}  status: {counts}")
    for rec in hits:
        print(f"HIT: {rec['status']} trial {rec['trial']} name={rec['config']['name']} "
              f"detail={rec.get('err', '')[:200]}")
    manifest.close()
    return 0 if not hits else 2


def run_trial_subprocess(cfg, args):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--device",
        str(args.device),
        "--config",
        json.dumps(cfg),
    ]
    env = dict(os.environ)
    if args.launch_blocking:
        env["ASCEND_LAUNCH_BLOCKING"] = "1"
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, start_new_session=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        out, _ = proc.communicate(timeout=args.timeout)
        status, rec = parse_child_output(out, proc.returncode, cfg)
    except subprocess.TimeoutExpired:
        kill_process_group(proc)
        out, _ = proc.communicate()
        status, rec = "HANG", {}
    rec = {"status": status, "config": cfg, "elapsed_s": round(time.time() - t0, 1), **rec}
    if status != "OK":
        rec["stdout_tail"] = out[-2000:]
    return rec


def parse_child_output(out, returncode, cfg):
    result = None
    for line in out.splitlines():
        if line.startswith("RESULT:"):
            result = json.loads(line[len("RESULT:"):])
    if result is not None:
        return result.get("status", "ERROR"), result
    # no RESULT line: child died hard (segfault, oom-kill, ...)
    return "ERROR", {"err": f"child returncode={returncode}", "grid_size": None}


# ----------------------------------------------------------------- child ----


def child_env_info():
    import torch
    import torch_npu
    import triton

    info = {
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "triton": triton.__version__,
        "device_count": torch.npu.device_count() if hasattr(torch, "npu") else 0,
    }
    try:
        info["device_name"] = torch.npu.get_device_properties(0).name
    except Exception:  # noqa: BLE001
        info["device_name"] = "unknown"
    print("RESULT:" + json.dumps(info))


def child_bg_load(args):
    import torch

    torch.npu.set_device(args.device)
    a = torch.randn(4096, 4096, device="npu", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="npu", dtype=torch.bfloat16)
    i = 0
    while True:
        c = a @ b
        i += 1
        if i % 8 == 0:
            torch.npu.synchronize()


def child_run_trial(cfg, args):
    import torch
    import torch_npu  # noqa: F401
    import triton

    import sgl_kernel_npu  # noqa: F401  registers npu ops
    from sgl_kernel_npu.fla.solve_tril import (
        merge_16x16_to_64x64_inverse_kernel,
        merge_16x16_to_64x64_inverse_kernel_reorder_all_masked,
        solve_tril_16x16_kernel_paral_v3,
    )
    from sgl_kernel_npu.fla.utils import prepare_chunk_indices

    dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

    dev = torch.device(f"npu:{args.device}")
    torch.npu.set_device(dev)
    torch.manual_seed(cfg["seed"])

    a_dtype = dtypes[cfg["a_dtype"]]
    out_dtype = dtypes[cfg["out_dtype"]]

    lens = cfg.get("lens")
    if lens is not None:
        B, T = 1, int(sum(lens))
        cu_seqlens = torch.tensor(
            [0] + [int(x) for x in torch.tensor(lens).cumsum(0)], dtype=torch.int64, device=dev
        )
    else:
        B, T = cfg["B"], cfg["T"]
        cu_seqlens = None
    H = cfg["H"]

    # A[b, t, h, :] is row (t % 64) of the strictly-lower 64x64 block, so
    # entry j must vanish for j > t % 64.
    r = (torch.arange(T, device=dev) % BT)
    keep = torch.arange(BT, device=dev)[None, :] <= r[:, None]  # [T, BT]
    A = torch.randn(B * T * H * BT, device=dev, dtype=torch.float32).reshape(B, T, H, BT)
    A = A * keep[None, :, None, :] * cfg["scale"]
    if a_dtype is not torch.float32:
        A = A.to(a_dtype)

    if cfg["contig"] == "lastdim":
        big = torch.zeros(B, T, H, BT * 2, device=dev, dtype=A.dtype)
        big[..., :BT] = A
        A_in = big[..., :BT]
    elif cfg["contig"] == "permute":
        A_in = A.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
    else:
        A_in = A

    ci1 = prepare_chunk_indices(cu_seqlens, LARGE_BLOCK_T) if cu_seqlens is not None else None
    NT1 = len(ci1) if cu_seqlens is not None else triton.cdiv(T, LARGE_BLOCK_T)
    ci2 = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT2 = len(ci2) if cu_seqlens is not None else triton.cdiv(T, BT)
    grid_size = NT2 * B * H

    print(f"[child] {cfg['name']}: B={B} T={T} H={H} varlen={cu_seqlens is not None} "
          f"grid={NT2}x{B * H}={grid_size} contig={cfg['contig']} "
          f"variant={cfg['variant']} w{cfg['num_warps']}/s{cfg['num_stages']} "
          f"repeat={cfg['repeat']}", flush=True)

    blocks = B * H * (triton.cdiv(T, BT) if cu_seqlens is None else NT2)
    do_check = (
        cfg["contig"] == "c"
        and cfg["scale"] <= 1.0
        and blocks <= CHECK_BLOCK_LIMIT
        and a_dtype == torch.float32
    )

    try:
        # stage 1: 16x16 diagonal-block inverses (production config)
        Ad = torch.empty(B, T, H, 16, device=dev, dtype=torch.float32)
        solve_tril_16x16_kernel_paral_v3[NT1, B * H](
            A=A_in,
            Ad=Ad,
            cu_seqlens=cu_seqlens,
            chunk_indices=ci1,
            T=T,
            H=H,
            BT=BT,
            LARGE_BLOCK_T=LARGE_BLOCK_T,
            num_warps=1,
            num_stages=4,
        )
        torch.npu.synchronize()

        # stage 2: the faulting merge kernel, with fuzzed launch knobs
        Ai = None
        for _ in range(max(1, cfg["repeat"])):
            Ai = torch.zeros(B, T, H, BT, device=dev, dtype=out_dtype)
            if cfg["variant"] == "reorder":
                merge_16x16_to_64x64_inverse_kernel_reorder_all_masked[
                    triton.cdiv(NT2, cfg["cpp"]), B * H
                ](
                    A=A_in,
                    Ad=Ad,
                    Ai=Ai,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=ci2,
                    T=T,
                    H=H,
                    BT=BT,
                    CHUNKS_PER_PROGRAM=cfg["cpp"],
                    NT=NT2,
                    num_warps=cfg["num_warps"],
                    num_stages=cfg["num_stages"],
                )
            else:
                merge_16x16_to_64x64_inverse_kernel[NT2, B * H](
                    A=A_in,
                    Ad=Ad,
                    Ai=Ai,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=ci2,
                    T=T,
                    H=H,
                    BT=BT,
                    num_warps=cfg["num_warps"],
                    num_stages=cfg["num_stages"],
                )
            torch.npu.synchronize()

        rec = {"status": "OK", "grid_size": grid_size, "blocks": blocks, "checked": do_check}
        if do_check:
            ref = torch_reference_inverse(A, cu_seqlens, B, T, H)
            max_diff = (ref - Ai.to(torch.float32)).abs().max().item()
            tol = 5e-3 if cfg["out_dtype"] == "fp32" else 3e-2
            rec["max_abs_diff"] = round(max_diff, 6)
            rec["tol"] = tol
            if max_diff > tol:
                rec["status"] = "MISMATCH"
        print("RESULT:" + json.dumps(rec))
        return EXIT_OK if rec["status"] == "OK" else EXIT_MISMATCH
    except RuntimeError as e:
        msg = str(e)
        rec = {"grid_size": grid_size, "err": msg[:1000]}
        if any(m.lower() in msg.lower() for m in TIMEOUT_MARKERS):
            rec["status"] = "AICORE_TIMEOUT"
            print("RESULT:" + json.dumps(rec))
            return EXIT_AICORE_TIMEOUT
        rec["status"] = "ERROR"
        print("RESULT:" + json.dumps(rec))
        return EXIT_ERROR
    except Exception as e:  # noqa: BLE001  e.g. triton CompilationError / OutOfResources
        import traceback

        rec = {
            "status": "ERROR",
            "grid_size": grid_size,
            "err": f"{type(e).__name__}: {e}",
            "traceback_tail": traceback.format_exc()[-1500:],
        }
        print("RESULT:" + json.dumps(rec))
        return EXIT_ERROR


def torch_reference_inverse(A, cu_seqlens, B, T, H):
    """Batched (I + A)^-1 per 64-token block, mirroring the fla semantics."""
    import torch

    def inv_bht(x):  # x: [b, h, t, BT]
        bb, hh, tt, bt = x.shape
        pad = (bt - tt % bt) % bt
        xp = torch.nn.functional.pad(x, (0, 0, 0, pad))
        blocks = xp.reshape(bb, hh, -1, bt, bt) + torch.eye(bt, device=x.device, dtype=x.dtype)
        return torch.linalg.inv(blocks).reshape(bb, hh, -1, bt)[:, :, :tt, :]

    if cu_seqlens is None:
        return inv_bht(A.permute(0, 2, 1, 3).to(torch.float32)).permute(0, 2, 1, 3)
    parts = []
    cl = cu_seqlens.tolist()
    for s, e in zip(cl[:-1], cl[1:]):
        if e > s:
            parts.append(inv_bht(A[0, s:e].transpose(0, 1)[None].to(torch.float32))[0].transpose(0, 1))
        else:
            parts.append(A[0, 0:0])
    return torch.cat(parts, dim=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=0, help="npu device index")
    ap.add_argument("--timeout", type=int, default=180, help="per-trial wall-clock timeout in seconds")
    ap.add_argument("--only", type=str, default=None, help="substring filter on scenario name")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1, help="multiply per-scenario launch repeat count")
    ap.add_argument("--soak-min", type=int, default=0, help="keep looping the corpus for N minutes")
    ap.add_argument("--max-trials", type=int, default=0, help="0 = unlimited (single corpus pass by default)")
    ap.add_argument("--launch-blocking", action="store_true", help="child runs with ASCEND_LAUNCH_BLOCKING=1")
    ap.add_argument("--bg-load", type=int, default=0, help="N background matmul workers for device pressure")
    ap.add_argument("--stop-on-hit", dest="stop_on_hit", action="store_true", default=True)
    ap.add_argument("--no-stop-on-hit", dest="stop_on_hit", action="store_false")
    ap.add_argument("--manifest", type=str, default="repro_solve_tril_manifest.jsonl")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--config", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--env-info", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--bg-load-worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.env_info:
        child_env_info()
        return
    if args.bg_load_worker:
        child_bg_load(args)
        return
    if args.child:
        if not args.config:
            print("RESULT:" + json.dumps({"status": "ERROR", "err": "--child requires --config"}))
            sys.exit(EXIT_ERROR)
        sys.exit(child_run_trial(json.loads(args.config), args))

    if args.list:
        for s in scenarios():
            if args.only and args.only not in s["name"]:
                continue
            lens = s["lens"]
            T = sum(lens) if lens else s["T"]
            print(f"{s['name']:<36} B={s['B']} T={T} H={s['H']} "
                  f"varlen={lens is not None} contig={s['contig']} "
                  f"variant={s['variant']} w{s['num_warps']}/s{s['num_stages']} repeat={s['repeat']}")
        return

    sys.exit(main_parent(args))


if __name__ == "__main__":
    main()
