"""
launch_quant.py — parallel quantization launcher
=================================================

Reads the JOB_TABLE below and spawns one `quant.py` subprocess per row,
each pinned to its own GPU via CUDA_VISIBLE_DEVICES.  All jobs run
concurrently; the launcher waits for every process to finish and prints
a final pass/fail summary.

Each job's stdout + stderr is tee'd to:
    LOG_ROOT/<module>/<opt_tag>/launch.log

Usage
-----
    # Run all jobs defined in JOB_TABLE:
    python quantization/launch_quant.py

    # Dry-run: print the commands that would be launched without running them:
    python quantization/launch_quant.py --dry-run

Job table format
----------------
Each row is a dict with keys:
    module      str   module name passed to quant.py --module
    optimized   int   0=oob | 1=sabre | 2=katana, passed to --optimized
    gpu         int   physical GPU id — becomes CUDA_VISIBLE_DEVICES=<gpu>
                      AND --gpu 0 inside the subprocess (device 0 of that
                      restricted environment)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Job table — edit this to define what runs on which GPU
# ---------------------------------------------------------------------------
# fmt: off
JOB_TABLE = [
    # module                                                      opt  gpu
    # Job 1: InternViT300M_Pixel_Unshuffle_MLP1 / katana (optimized=2)
    {"module": "InternViT300M_Pixel_Unshuffle_MLP1",             "optimized": 0, "gpu": 0},
    # {"module": "InternViT300M_Pixel_Unshuffle_MLP1",             "optimized": 1, "gpu": 0},
    {"module": "InternViT300M_Pixel_Unshuffle_MLP1",             "optimized": 2, "gpu": 0},
    # Job 2: nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA / sabre (optimized=1)
    {"module": "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA", "optimized": 0, "gpu": 1},
    # {"module": "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA", "optimized": 1, "gpu": 1},
    {"module": "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA", "optimized": 2, "gpu": 1},
]
# fmt: on

# ---------------------------------------------------------------------------
# Parallel sensitivity configuration
# ---------------------------------------------------------------------------
# When PARALLEL_SENSITIVITY = True:
#   • Jobs run ONE AT A TIME (sequential) so each job gets the full GPU pool
#     for its sensitivity scan workers.
#   • SENSITIVITY_GPU_POOL lists ALL physical GPU IDs available for workers.
#     Workers are assigned round-robin across this pool.
#   • VRAM_PER_GPU_GB / CUSHION_GB drive the allocation formula.
#
# Formula (spec-correct):
#   U               = m_u + sn_per_worker   (total memory unit per worker)
#   currgpu_avail_g = M - O_g               (live-polled via pynvml)
#   workers_g       = floor(currgpu_avail_g / U)
#   P               = sum(workers_g for g in SENSITIVITY_GPU_POOL)
#
# CUSHION_GB approximates O when pynvml is unavailable (static fallback).
# SN_PER_WORKER_GB is the per-worker safety net (spec: sn).  It is looked
# up from ARCHITECTURE_SAFETY_NET_GB first; DEFAULT_SN_PER_WORKER_GB is
# used for any module not listed there.
#
# When PARALLEL_SENSITIVITY = False (default):
#   • Original concurrent launch behaviour is unchanged.
#   • No sensitivity flags are passed to quant.py.
# ---------------------------------------------------------------------------
PARALLEL_SENSITIVITY  = True           # jobs run sequentially; each gets full GPU pool
DYNAMIC_SCHEDULING    = True           # use Ray-based dynamic scheduler (requires ray[default])
                                       # set False to fall back to static subprocess workers
SENSITIVITY_GPU_POOL  = [0, 1, 2, 3]  # GPUs available for sensitivity-scan workers
VRAM_PER_GPU_GB       = 32.0           # M: VRAM per GPU in GB
CUSHION_GB            = 3.0            # O fallback: reserved per GPU for OS/driver safety
GPU_HEADROOM_GB       = 0.5            # per-GPU global VRAM headroom (GB) deducted before
                                       # computing workers_g:
                                       #   effective_avail = currgpu_avail - GPU_HEADROOM_GB
                                       #   workers_g = floor(effective_avail / U)
                                       # Covers driver overhead, fragmentation, and processes
                                       # that start after the pynvml poll.

# ---------------------------------------------------------------------------
# Per-architecture safety net (sn) calibration
# ---------------------------------------------------------------------------
# sn is the per-worker memory buffer added to m_u to form U = m_u + sn.
# It absorbs activation spikes, cuDNN workspace growth, and BFCArena
# fragmentation that the dry-run probe (n_warmup=3) does not capture.
#
# Calibration procedure:
#   1. Run estimate_vram_gb(n_warmup=3)  → K_warmup  (= m_u from probe)
#   2. Run estimate_vram_gb(n_warmup=50) → K_full    (full eval pass)
#   3. sn = K_full - K_warmup + 0.5 GB  (0.5 GB driver/fragmentation overhead)
#
# Values below are empirically calibrated for this codebase:
#   InternViT300M_Pixel_Unshuffle_MLP1:
#       Observed actual per-worker VRAM = 2.37 GB (raw probe, oob run).
#       Probe with 1.10× multiplier → m_u = 2.61 GB.
#       Actual peak never exceeded raw probe value during full scan, so
#       sn = 0.5 GB (driver/fragmentation overhead only) is sufficient.
#       U = 2.61 + 0.50 = 3.11 GB → 9 workers/GPU (vs 6 with sn=1.5).
#   nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA:
#       Larger LLM backbone with longer activation buffers.  sn = 2.5 GB.
# ---------------------------------------------------------------------------
ARCHITECTURE_SAFETY_NET_GB: dict[str, float] = {
    "InternViT300M_Pixel_Unshuffle_MLP1":             0.5,  # reduced from 1.5; actual peak ≈ raw probe (2.37 GB)
    "nCoT_Token_Interleaver_Qwen2_05_RouteSpeed_VLA": 2.5,
}
DEFAULT_SN_PER_WORKER_GB: float = 2.0   # fallback for any unlisted module

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
QUANT_SCRIPT = Path(__file__).parent / "quant.py"
LOG_ROOT     = Path("/local/mnt/workspace/users/sathya/projects/simlingo/quantization/logs")

OPT_TAGS = {0: "oob", 1: "sabre", 2: "katana"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class Job(NamedTuple):
    module:    str
    optimized: int
    gpu:       int

    @property
    def opt_tag(self) -> str:
        return OPT_TAGS[self.optimized]

    @property
    def label(self) -> str:
        return f"{self.module}/{self.opt_tag}@gpu{self.gpu}"

    @property
    def log_path(self) -> Path:
        log_dir = LOG_ROOT / self.module / self.opt_tag
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "launch.log"


def _resolve_sn_per_worker(module: str) -> float:
    """Return the per-worker safety net (sn) in GB for *module*.

    Looks up ARCHITECTURE_SAFETY_NET_GB first; falls back to
    DEFAULT_SN_PER_WORKER_GB for any module not listed there.
    """
    return ARCHITECTURE_SAFETY_NET_GB.get(module, DEFAULT_SN_PER_WORKER_GB)


def _build_cmd(job: Job) -> list[str]:
    """Return the argv list for the quant.py subprocess.

    When PARALLEL_SENSITIVITY is True, appends the sensitivity flags so
    quant.py knows to use run_parallel_sensitivity() instead of the sequential
    AIMET scan.  The per-architecture sn_per_worker value is resolved from
    ARCHITECTURE_SAFETY_NET_GB and passed via --sn-per-worker.
    """
    cmd = [
        sys.executable,          # same python interpreter as the launcher
        str(QUANT_SCRIPT),
        "--module",    job.module,
        "--optimized", str(job.optimized),
        "--gpu",       "0",       # always device 0 inside the restricted env
    ]
    if PARALLEL_SENSITIVITY:
        sn = _resolve_sn_per_worker(job.module)
        cmd += [
            "--parallel-sensitivity",
            "--sensitivity-gpu-ids", ",".join(str(g) for g in SENSITIVITY_GPU_POOL),
            "--vram-per-gpu",        str(VRAM_PER_GPU_GB),
            "--cushion-gb",          str(CUSHION_GB),
            "--sn-per-worker",       str(sn),
            "--gpu-headroom-gb",     str(GPU_HEADROOM_GB),
        ]
        if DYNAMIC_SCHEDULING:
            cmd.append("--dynamic-scheduling")
    return cmd


def _build_env(job: Job) -> dict[str, str]:
    """Return an env dict with CUDA_VISIBLE_DEVICES set to the physical GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    return env


# ---------------------------------------------------------------------------
# Launcher — concurrent (original) and sequential (parallel-sensitivity) modes
# ---------------------------------------------------------------------------

def _run_one_job(job: Job, col: int, start_time: float) -> int:
    """Launch *job*, wait for it to finish, return its exit code.

    Used by _launch_sequential so each job runs to completion before the
    next one starts — ensuring the full GPU pool is free for sensitivity workers.
    """
    cmd      = _build_cmd(job)
    env      = _build_env(job)
    log_path = job.log_path
    cmd_str  = " ".join(cmd)

    print(f"  [{job.label:<{col}}]  log → {log_path}")

    log_fh = log_path.open("w", encoding="utf-8")
    log_fh.write(f"# CMD : CUDA_VISIBLE_DEVICES={job.gpu} {cmd_str}\n")
    log_fh.write(f"# START: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    log_fh.flush()

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Poll with heartbeat while this single job runs
    last_beat     = time.monotonic()
    BEAT_INTERVAL = 60

    while True:
        rc = proc.poll()
        if rc is not None:
            break
        now = time.monotonic()
        if now - last_beat >= BEAT_INTERVAL:
            elapsed = now - start_time
            print(f"  [heartbeat {elapsed:.0f}s]  still running: {job.label}")
            last_beat = now
        time.sleep(2)

    log_fh.write(f"\n# END  : {time.strftime('%Y-%m-%d %H:%M:%S')}  rc={rc}\n")
    log_fh.close()

    elapsed = time.monotonic() - start_time
    status  = "OK" if rc == 0 else f"FAILED (rc={rc})"
    print(f"  [{job.label:<{col}}]  {status}  ({elapsed:.0f}s elapsed)")
    return rc


def _launch_sequential(jobs: list[Job], dry_run: bool = False) -> None:
    """Run jobs one at a time so each gets the full GPU pool for sensitivity workers.

    Used when PARALLEL_SENSITIVITY = True.  The total wall-clock time is the
    sum of all jobs, but the sensitivity scan within each job is parallelised
    across all GPUs, so the net speedup is still large.
    """
    col = 60

    print(f"\n{'='*72}")
    sched_mode = "Ray dynamic" if DYNAMIC_SCHEDULING else "static subprocess"
    print(f"  Launching {len(jobs)} job(s) SEQUENTIALLY (PARALLEL_SENSITIVITY=True, scheduler={sched_mode})")
    print(f"  Sensitivity GPU pool: {SENSITIVITY_GPU_POOL}")
    print(f"  VRAM per GPU: {VRAM_PER_GPU_GB} GB  |  Cushion (O fallback): {CUSHION_GB} GB  |  GPU headroom: {GPU_HEADROOM_GB} GB")
    print(f"  Per-architecture sn: {ARCHITECTURE_SAFETY_NET_GB}  |  default: {DEFAULT_SN_PER_WORKER_GB} GB")
    print(f"{'='*72}\n")

    if dry_run:
        for job in jobs:
            cmd     = _build_cmd(job)
            cmd_str = " ".join(cmd)
            print(f"  [{job.label:<{col}}]  log → {job.log_path}")
            print(f"    CMD : CUDA_VISIBLE_DEVICES={job.gpu} {cmd_str}\n")
        print("Dry-run complete — no processes were started.\n")
        return

    start_time = time.monotonic()
    finished: list[tuple[Job, int]] = []

    for job in jobs:
        rc = _run_one_job(job, col, start_time)
        finished.append((job, rc))

    total_elapsed = time.monotonic() - start_time
    n_ok   = sum(1 for _, rc in finished if rc == 0)
    n_fail = len(finished) - n_ok

    print(f"\n{'='*72}")
    print(f"  Summary: {n_ok}/{len(finished)} succeeded  |  {n_fail} failed  |  {total_elapsed:.0f}s total")
    print(f"{'='*72}\n")

    if n_fail:
        print("  Failed jobs:")
        for job, rc in finished:
            if rc != 0:
                print(f"    {job.label}  (rc={rc})  log: {job.log_path}")
        print()
        sys.exit(1)


def _launch_concurrent(jobs: list[Job], dry_run: bool = False) -> None:
    """Original concurrent launch — one subprocess per job, all running in parallel.

    Used when PARALLEL_SENSITIVITY = False (default).  Behaviour is identical
    to the original launch_all() implementation.
    """
    col = 60

    print(f"\n{'='*72}")
    print(f"  Launching {len(jobs)} quantization job(s) concurrently")
    print(f"{'='*72}\n")

    procs: list[tuple[Job, subprocess.Popen, object]] = []

    for job in jobs:
        cmd = _build_cmd(job)
        env = _build_env(job)
        log_path = job.log_path

        cmd_str = " ".join(cmd)
        print(f"  [{job.label:<{col}}]  log → {log_path}")
        if dry_run:
            print(f"    CMD : CUDA_VISIBLE_DEVICES={job.gpu} {cmd_str}\n")
            continue

        log_fh = log_path.open("w", encoding="utf-8")
        log_fh.write(f"# CMD : CUDA_VISIBLE_DEVICES={job.gpu} {cmd_str}\n")
        log_fh.write(f"# START: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_fh.flush()

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs.append((job, proc, log_fh))

    if dry_run:
        print("\nDry-run complete — no processes were started.\n")
        return

    print(f"\n  All {len(procs)} process(es) started.  Waiting for completion...\n")

    start_time    = time.monotonic()
    last_beat     = start_time
    BEAT_INTERVAL = 60

    remaining = list(procs)
    finished:  list[tuple[Job, int]] = []

    while remaining:
        still_running = []
        for job, proc, log_fh in remaining:
            rc = proc.poll()
            if rc is None:
                still_running.append((job, proc, log_fh))
            else:
                log_fh.write(f"\n# END  : {time.strftime('%Y-%m-%d %H:%M:%S')}  rc={rc}\n")
                log_fh.close()
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                elapsed = time.monotonic() - start_time
                print(f"  [{job.label:<{col}}]  {status}  ({elapsed:.0f}s elapsed)")
                finished.append((job, rc))

        remaining = still_running

        if remaining:
            now = time.monotonic()
            if now - last_beat >= BEAT_INTERVAL:
                elapsed = now - start_time
                labels  = ", ".join(j.label for j, _, _ in remaining)
                print(f"  [heartbeat {elapsed:.0f}s]  still running: {labels}")
                last_beat = now
            time.sleep(2)

    total_elapsed = time.monotonic() - start_time
    n_ok   = sum(1 for _, rc in finished if rc == 0)
    n_fail = len(finished) - n_ok

    print(f"\n{'='*72}")
    print(f"  Summary: {n_ok}/{len(finished)} succeeded  |  {n_fail} failed  |  {total_elapsed:.0f}s total")
    print(f"{'='*72}\n")

    if n_fail:
        print("  Failed jobs:")
        for job, rc in finished:
            if rc != 0:
                print(f"    {job.label}  (rc={rc})  log: {job.log_path}")
        print()
        sys.exit(1)


def launch_all(jobs: list[Job], dry_run: bool = False) -> None:
    """Dispatch to sequential or concurrent launcher based on PARALLEL_SENSITIVITY."""
    if PARALLEL_SENSITIVITY:
        _launch_sequential(jobs, dry_run)
    else:
        _launch_concurrent(jobs, dry_run)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel quantization launcher")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without launching any processes",
    )
    parser.add_argument(
        "--dynamic-scheduling", action=argparse.BooleanOptionalAction, default=None,
        help="Enable (--dynamic-scheduling) or disable (--no-dynamic-scheduling) "
             "Ray-based dynamic scheduler.  "
             f"Overrides the DYNAMIC_SCHEDULING constant (default: {DYNAMIC_SCHEDULING}).",
    )
    parser.add_argument(
        "--gpu-headroom-gb", type=float, default=None,
        help="Per-GPU global VRAM headroom in GB deducted before computing workers_g "
             "(effective_avail = currgpu_avail - gpu_headroom_gb).  "
             f"Overrides the GPU_HEADROOM_GB constant (default: {GPU_HEADROOM_GB} GB).",
    )
    parser.add_argument(
        "--sn-per-worker", type=float, default=None,
        help="Override the per-worker safety net (sn) in GB for ALL modules, "
             "bypassing ARCHITECTURE_SAFETY_NET_GB lookup.  "
             f"Default: per-architecture table (fallback {DEFAULT_SN_PER_WORKER_GB} GB).",
    )
    parser.add_argument(
        "--vram-per-gpu", type=float, default=None,
        help=f"Override VRAM_PER_GPU_GB (default: {VRAM_PER_GPU_GB} GB).",
    )
    parser.add_argument(
        "--cushion-gb", type=float, default=None,
        help=f"Override CUSHION_GB static fallback (default: {CUSHION_GB} GB).",
    )
    args = parser.parse_args()

    # Apply CLI overrides to module-level constants so _build_cmd() picks them up.
    if getattr(args, "dynamic_scheduling", None) is not None:
        DYNAMIC_SCHEDULING = args.dynamic_scheduling
    if args.gpu_headroom_gb is not None:
        GPU_HEADROOM_GB = args.gpu_headroom_gb
    if args.vram_per_gpu is not None:
        VRAM_PER_GPU_GB = args.vram_per_gpu
    if args.cushion_gb is not None:
        CUSHION_GB = args.cushion_gb

    # Per-worker sn override: replace the architecture table with a uniform value.
    if args.sn_per_worker is not None:
        _sn_override = args.sn_per_worker
        _resolve_sn_per_worker = lambda module: _sn_override  # noqa: E731

    jobs = [Job(**row) for row in JOB_TABLE]
    launch_all(jobs, dry_run=args.dry_run)
