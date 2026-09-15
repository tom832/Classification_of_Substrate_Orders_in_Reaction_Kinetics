#!/usr/bin/env python3
"""Generate unlabeled ODE pools for prepare_data_tls_pipeline.ipynb.

Examples (run in an environment with numpy/scipy):
  python solve_ivp_synthetic.py --class_num 4 --job_num 600000 --num_proc 24
  python solve_ivp_synthetic.py --class_num 5 --job_num 400000 --num_proc 24

4 creates the no-k4 pool; 5 creates the side-reaction pool used ONLY for S5.
The five-class notebook also needs the no-k4 pool for S1-S4.
All five cycle constants are independently nonzero (one reversible topology).
Active constants use uniform(1,10)*10**choice(range(-5,5)); no TLS filtering
or targeted side-reaction prior is applied during generation.

Default substrate accounting includes BOTH substrate-consuming steps, as in
the response-letter ODE. --mass_mode legacy is available explicitly and is
recorded; the notebook rejects mixing accounting conventions.

--job_num counts accepted trajectories, --log_interval counts processed
parameter proposals. Output: data/ode_raw_data/{4,5}_class/*.json, one file
per trajectory. Rerun with a larger job_num to extend the same seeded pool;
another seed adds independent samples in the same directory. job_num counts
accepted records for the CURRENT seed/config, including previously saved ones.
Parallel results are committed in proposal order: changing num_proc does not
change the accepted samples. Workers and completed-result buffers are bounded.
"""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import hashlib
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import random
import sys
import tempfile
import time

# Each process runs one LSODA integration; avoid nested BLAS parallelism.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np
from scipy.integrate import solve_ivp

SCHEMA = "cosork-unified-ode-v1"
RATE_NAMES = ("k1", "km1", "k2", "km2", "k3", "k4")


def mechanism(t, z, rates, strict_mass=True):
    """State: [S, P, Cat, CatS, CatSprime, BP]; always six species."""
    s, p, cat, cats, catsp, bp = z
    k1, km1, k2, km2, k3, k4 = rates
    binding = k1 * cat * s - km1 * cats
    conversion = k2 * cats - km2 * catsp
    turnover = k3 * catsp * s
    side = k4 * s
    return (-binding - side - (turnover if strict_mass else 0.0),
            turnover, -binding + turnover, binding - conversion,
            conversion - turnover, side)


@contextmanager
def quiet_solver():
    """Capture native LSODA diagnostics without flooding HPC logs."""
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        os.dup2(null, 2)
        yield
    finally:
        for fd, old in zip((1, 2), saved):
            os.dup2(old, fd)
            os.close(old)
        os.close(null)


def sample_rates(seed, idx, class_num):
    rng = random.Random(f"cosork-ode-v1:{seed}:{idx}")
    rates = [rng.uniform(1, 10) * 10 ** rng.choice(range(-5, 5))
             for _ in range(6)]
    if class_num == 4:
        rates[-1] = 0.0
    return rates


def solve_task(idx, cfg):
    rates = sample_rates(cfg["seed"], idx, cfg["class_num"])
    calls = 0

    def rhs(t, z):
        nonlocal calls
        calls += 1
        if calls > cfg["max_rhs_evals"]:
            raise RuntimeError("rhs_limit")
        return mechanism(t, z, rates, cfg["mass_mode"] == "strict")

    t = np.linspace(0, cfg["t_end"], cfg["n_points"])
    try:
        with quiet_solver():
            sol = solve_ivp(rhs, (0, cfg["t_end"]),
                            [1.0, 0.0, cfg["cat0"], 0.0, 0.0, 0.0],
                            method="LSODA", t_eval=t,
                            rtol=cfg["rtol"], atol=cfg["atol"])
    except RuntimeError as exc:
        if str(exc) == "rhs_limit":
            return idx, "rhs_limit", None
        raise
    if not sol.success or sol.y.shape != (6, len(t)):
        return idx, "solver_fail", None
    z = sol.y
    if not np.isfinite(z).all():
        return idx, "non_finite", None
    if z.min() < -cfg["negative_tol"]:
        return idx, "negative", None
    # Check the invariants before clipping numerical undershoots.
    cat_error = float(np.max(np.abs(z[2:5].sum(axis=0) - cfg["cat0"])))
    p_factor = 2 if cfg["mass_mode"] == "strict" else 1
    mass_error = float(np.max(np.abs(z[0] + p_factor*z[1] + z[3] + z[4] + z[5] - 1)))
    if max(cat_error, mass_error) > cfg["mass_tol"]:
        return idx, "mass_balance", None
    z = np.maximum(z, 0.0)
    s = z[0]
    if s[cfg["guard_idx"]] <= cfg["s_guard_min"] or s[-1] <= 0:
        return idx, "too_fast", None
    if s[-1] >= cfg["final_max"]:
        return idx, "too_slow", None
    record = dict(schema=SCHEMA, raw_id=f"{cfg['pool_id']}:{idx}", idx=idx,
                  pool_kind="cycle" if cfg["class_num"] == 4 else "side",
                  s0=1.0, cat0=cfg["cat0"], mass_mode=cfg["mass_mode"],
                  t=t.tolist(), nfev=calls,
                  catalyst_balance_error=cat_error, substrate_balance_error=mass_error)
    record['generation_config'] = cfg
    record.update(zip(RATE_NAMES, rates))
    record.update({key: values.tolist() for key, values in
                   zip(("s", "p", "cat", "cats", "catsp", "bp"), z)})
    return idx, "accepted", record


def duration(seconds):
    if not math.isfinite(seconds):
        return "unknown"
    seconds = int(max(0, seconds))
    return f"{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}"


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--class_num", type=int, choices=(4, 5), required=True)
    p.add_argument("--job_num", type=int, default=200000)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=Path,
                   help="default data/ode_raw_data/{4,5}_class; one JSON per trajectory")
    p.add_argument("--log_interval", type=int, default=10000,
                   help="one progress line per this many PROCESSED proposals")
    p.add_argument("--max_attempts", type=int, default=0,
                   help="0 = unlimited; positive = stop after this many proposals")
    p.add_argument("--mass_mode", choices=("strict", "legacy"), default="strict")
    p.add_argument("--t_end", type=float, default=180.0)
    p.add_argument("--n_points", type=int, default=30)
    p.add_argument("--cat0", type=float, default=0.01)
    p.add_argument("--s_guard_min", type=float, default=0.05)
    p.add_argument("--final_max", type=float, default=0.5)
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-9)
    p.add_argument("--max_rhs_evals", type=int, default=50000)
    p.add_argument("--negative_tol", type=float, default=1e-6)
    p.add_argument("--mass_tol", type=float, default=1e-6)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    for key in ("job_num", "num_proc", "log_interval", "max_rhs_evals"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive")
    if args.n_points < 3 or args.max_attempts < 0 or args.seed < 0:
        parser.error("n_points >= 3, max_attempts >= 0 and seed >= 0 required")
    for key in ("t_end", "cat0", "rtol", "atol", "negative_tol", "mass_tol"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key} must be finite and positive")
    if not 0 < args.s_guard_min < 1 or not 0 < args.final_max < 1:
        parser.error("guard and final_max must lie in (0,1)")
    cfg = {k: v for k, v in vars(args).items()
           if k not in ("out_dir", "num_proc", "job_num", "log_interval", "max_attempts")}
    cfg["guard_idx"] = round(19 / 29 * (args.n_points - 1))
    identity = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:20]
    cfg["pool_id"] = f"{args.class_num}c-{identity}"
    out = (args.out_dir or Path(__file__).resolve().parent / "data" /
           "ode_raw_data" / f"{args.class_num}_class").resolve()
    out.mkdir(parents=True, exist_ok=True)
    prefix = cfg['pool_id'] + '-'
    existing = list(out.glob(prefix + '*.json'))
    initial_saved = len(existing)
    next_idx = max((int(p.stem[len(prefix):]) for p in existing), default=-1) + 1
    logger = logging.getLogger("ode_pipeline")
    logger.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(out / "generation.log")):
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(h)
    logger.info('config=%s', json.dumps(cfg, sort_keys=True))
    if initial_saved >= args.job_num:
        logger.info('Already saved %d/%d for this seed/config; nothing to generate.',
                    initial_saved, args.job_num)
        return 0
    start = time.monotonic()
    saved = initial_saved
    processed = proposed = next_idx
    reasons = Counter()
    active, ready = {}, {}
    status = "running"
    cap = args.num_proc * 4

    def report():
        wall = time.monotonic() - start
        added = saved - initial_saved
        eta = wall * (args.job_num - saved) / added if added else math.inf
        logger.info("saved=%d/%d processed_this_run=%d accepted=%.3f%% elapsed=%s ETA=%s counts=%s",
                    saved, args.job_num, processed-next_idx, 100*added/max(1, processed-next_idx),
                    duration(wall), duration(eta), dict(reasons))

    logger.info("start pool=%s target=%d workers=%d mass_mode=%s log_interval=%d proposals",
                cfg["pool_id"], args.job_num, args.num_proc, args.mass_mode, args.log_interval)
    executor = ProcessPoolExecutor(max_workers=args.num_proc,
                                   mp_context=multiprocessing.get_context("spawn"))
    try:
        while saved < args.job_num:
            while proposed - processed < cap and (not args.max_attempts or proposed-next_idx < args.max_attempts):
                active[executor.submit(solve_task, proposed, cfg)] = proposed
                proposed += 1
            if not active and not ready:
                status = "insufficient_attempts"
                break
            if processed not in ready:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    expected = active.pop(future)
                    result = future.result()
                    if result[0] != expected:
                        raise RuntimeError("Worker proposal ID mismatch")
                    ready[expected] = result
            while processed in ready and saved < args.job_num:
                idx, reason, record = ready.pop(processed)
                processed += 1
                reasons[reason] += 1
                if record is not None:
                    target = out / f"{prefix}{idx:09d}.json"
                    with tempfile.NamedTemporaryFile(mode='w', dir=out,
                                                     prefix='.', suffix='.partial') as f:
                        json.dump(record, f, separators=(',', ':'), allow_nan=False)
                        f.write('\n')
                        f.flush()
                        # Publish only a complete JSON; never overwrite another file.
                        os.link(f.name, target)
                    saved += 1
                if (processed-next_idx) % args.log_interval == 0:
                    report()
        if saved == args.job_num:
            status = "complete"
    except KeyboardInterrupt:
        status = "interrupted"
        logger.warning("Interrupted; accepted records retained in %s", out)
    except BaseException:
        status = "failed"
        logger.exception("Generation failed")
        raise
    finally:
        for future in active:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        report()
        logger.info("%s output=%s", status, out)
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
