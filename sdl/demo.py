"""One-command demo: the closed loop searching for a better design.

Runs random / Bayesian / RL / hybrid optimizers on the perovskite domain under
identical budgets and prints a head-to-head comparison + the best design found.

    python -m sdl.demo
    python -m sdl.demo --domain branin --iterations 15
"""

from __future__ import annotations

import argparse
import time

from sdl.core.orchestrator import CampaignConfig, Orchestrator
from sdl.core.registry import get_domain, get_executor, get_optimizer


def run_one(domain_name, opt_name, iterations, batch, initial, noise, seed):
    domain = get_domain(domain_name)()
    optimizer = get_optimizer(opt_name)(domain.parameter_space(), domain.objectives(), seed=seed)
    executor = get_executor("simulation")(domain, noise=noise, seed=seed)
    cfg = CampaignConfig(n_iterations=iterations, batch_size=batch, n_initial=initial, seed=seed)
    t0 = time.perf_counter()
    campaign = Orchestrator(domain, optimizer, executor, cfg).run()
    return campaign, time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SDL demo: compare optimizers on one domain")
    ap.add_argument("--domain", default="reaction")
    ap.add_argument("--optimizers", nargs="+", default=["random", "bayesian", "physics_bo", "rl", "hybrid"])
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--initial", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    domain = get_domain(args.domain)()
    obj_names = domain.objectives().names

    print(f"\n  SELF-DRIVING LAB - autonomous discovery on '{args.domain}'")
    print(f"  budget: {args.initial} seed + {args.iterations}x{args.batch} = "
          f"{args.initial + args.iterations * args.batch} experiments each\n")
    header = f"  {'optimizer':<12} {'best_scalar':>12} " + " ".join(f"{n:>10}" for n in obj_names) + f" {'sec':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    best_overall = None
    for opt_name in args.optimizers:
        try:
            campaign, secs = run_one(args.domain, opt_name, args.iterations, args.batch,
                                     args.initial, args.noise, args.seed)
        except Exception as e:
            print(f"  {opt_name:<12} ERROR: {e}")
            continue
        m = campaign.best_result.metrics if campaign.best_result else {}
        metric_cells = " ".join(f"{m.get(n, float('nan')):>10.3f}" for n in obj_names)
        print(f"  {opt_name:<12} {campaign.best_scalar:>12.4f} {metric_cells} {secs:>7.1f}")
        if best_overall is None or campaign.best_scalar > best_overall[1]:
            best_overall = (opt_name, campaign.best_scalar, campaign.best_result)

    if best_overall and best_overall[2]:
        print(f"\n  Best design found by '{best_overall[0]}':")
        for k, v in best_overall[2].params.items():
            print(f"    {k:<14} = {v:.4f}" if isinstance(v, float) else f"    {k:<14} = {v}")
        print("  Predicted performance:")
        for k, v in best_overall[2].metrics.items():
            print(f"    {k:<14} = {v:.4f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
