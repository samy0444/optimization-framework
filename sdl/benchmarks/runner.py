"""Benchmark runner: optimizer x domain x seed, with recorded convergence.

Compares optimizers (e.g. BO+RL vs BO, RL and random) under identical
experiment budgets. Results are saved as JSON (via
BenchmarkRecorder) and summarized as mean +/- std of the best objective, plus
'experiments-to-target' (sample efficiency).

Usage:
    python -m sdl.benchmarks.runner --domain perovskite \
        --optimizers random bayesian rl hybrid --seeds 0 1 2 \
        --iterations 20 --batch 4 --out results/bench_perovskite.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import List, Optional

from sdl.core.orchestrator import CampaignConfig, Orchestrator
from sdl.core.registry import get_domain, get_executor, get_optimizer
from sdl.core.store import BenchmarkRecorder

logger = logging.getLogger("sdl.benchmarks")


def experiments_to_target(domain, convergence: List[float], batch: int, n_initial: int) -> Optional[int]:
    """Number of experiments until all objective targets were first met.

    Approximated from the scalar convergence trace crossing the target scalar.
    Returns None if never reached.
    """
    objs = domain.objectives()
    target_metrics = {o.name: o.target for o in objs.objectives if o.target is not None}
    if not target_metrics:
        return None
    # scalar value if every target is exactly met (upper bound on needed scalar)
    objs.update_scales([target_metrics])
    target_scalar = objs.scalarize(target_metrics)
    for i, best in enumerate(convergence):
        if best >= target_scalar:
            return i + 1
    return None


def run_benchmark(
    domain_name: str,
    optimizer_names: List[str],
    seeds: List[int],
    iterations: int,
    batch: int,
    n_initial: int,
    noise: float,
    out_path: str,
):
    recorder = BenchmarkRecorder(out_path)
    for opt_name in optimizer_names:
        for seed in seeds:
            domain = get_domain(domain_name)()
            optimizer = get_optimizer(opt_name)(domain.parameter_space(), domain.objectives(), seed=seed)
            executor = get_executor("simulation")(domain, noise=noise, seed=seed)
            config = CampaignConfig(
                n_iterations=iterations, batch_size=batch, n_initial=n_initial, seed=seed
            )
            t0 = time.perf_counter()
            campaign = Orchestrator(domain, optimizer, executor, config).run()
            wall = time.perf_counter() - t0
            ett = experiments_to_target(domain, campaign.convergence_history, batch, n_initial)
            recorder.add_run(
                domain=domain_name,
                optimizer=opt_name,
                seed=seed,
                convergence=campaign.convergence_history,
                best_scalar=campaign.best_scalar,
                best_metrics=campaign.best_result.metrics if campaign.best_result else None,
                n_experiments=campaign.n_experiments,
                wall_time_s=wall,
                extra={"experiments_to_target": ett},
            )
            logger.info("[%s seed=%d] best_scalar=%.4f ett=%s", opt_name, seed, campaign.best_scalar, ett)
    recorder.save()
    return recorder


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SDL optimizer benchmark")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--optimizers", nargs="+", default=["random", "bayesian", "rl", "hybrid"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--initial", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    recorder = run_benchmark(
        args.domain, args.optimizers, args.seeds, args.iterations,
        args.batch, args.initial, args.noise, args.out,
    )
    print("\n=== Benchmark summary (mean +/- std of best scalar) ===")
    print(json.dumps(recorder.summary(), indent=2))
    print(f"\nFull results saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
