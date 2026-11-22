"""Command-line entry point for the self-driving lab framework.

Examples:
    python -m sdl.cli list
    python -m sdl.cli run --domain branin --optimizer bayesian --iterations 20
    python -m sdl.cli run --domain branin --optimizer random --iterations 20 --seed 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from sdl.core.orchestrator import CampaignConfig, Orchestrator
from sdl.core.registry import available, get_domain, get_executor, get_optimizer


def _build(args):
    domain = get_domain(args.domain)()
    optimizer = get_optimizer(args.optimizer)(domain.parameter_space(), domain.objectives(), seed=args.seed)
    executor = get_executor(args.executor)(domain, noise=args.noise, seed=args.seed)
    config = CampaignConfig(
        n_iterations=args.iterations,
        batch_size=args.batch,
        n_initial=args.initial,
        n_validation=args.validation,
        seed=args.seed,
        store_root=args.store,
    )
    return domain, optimizer, executor, config


def cmd_list(_args) -> int:
    print(json.dumps(available(), indent=2))
    return 0


def cmd_run(args) -> int:
    domain, optimizer, executor, config = _build(args)
    t0 = time.perf_counter()
    campaign = Orchestrator(domain, optimizer, executor, config).run()
    wall = time.perf_counter() - t0

    summary = campaign.summary()
    summary["wall_time_s"] = round(wall, 3)
    print("\n=== Campaign summary ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sdl", description="Domain-agnostic self-driving lab")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available domains/optimizers/executors").set_defaults(func=cmd_list)

    run = sub.add_parser("run", help="Run an optimization campaign")
    run.add_argument("--domain", required=True)
    run.add_argument("--optimizer", default="bayesian")
    run.add_argument("--executor", default="simulation")
    run.add_argument("--iterations", type=int, default=20)
    run.add_argument("--batch", type=int, default=4)
    run.add_argument("--initial", type=int, default=8)
    run.add_argument("--validation", type=int, default=0)
    run.add_argument("--noise", type=float, default=0.0)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--store", default=None, help="Directory to persist results")
    run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
