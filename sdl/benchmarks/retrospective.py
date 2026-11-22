"""Tier-1 retrospective validation on a REAL public dataset.

Replays each optimizer over the Science 2018 Buchwald-Hartwig HTE dataset:
the optimizer proposes which reactions to "run", and we reveal only the real
measured yield of its picks. We measure how many experiments each method needs
to find a high-yield reaction - the sample-efficiency claim, on real chemistry.

Two scenarios:
  * default          - random vs standard BO vs physics_bo (no prior). Shows the
                       *engine's* efficiency on real data.
  * --prior          - physics_bo additionally warm-starts from a prior fit on a
                       random historical slice of the table (warm-start demo).

Usage:
  python -m sdl.benchmarks.retrospective --seeds 20 --out results/retro_buchwald.json
  python -m sdl.benchmarks.retrospective --seeds 20 --prior --out results/retro_buchwald_prior.json
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import mean, median
from typing import List, Optional

import numpy as np

from sdl.core.orchestrator import CampaignConfig, Orchestrator
from sdl.core.registry import get_domain, get_optimizer
from sdl.executors.tabular import TabularExecutor


def experiments_to_target(campaign, target_name: str, target_value: float) -> Optional[int]:
    """1-based index of the first experiment whose real measurement >= target."""
    for i, r in enumerate(campaign.results, start=1):
        if r.success and r.metrics.get(target_name, float("-inf")) >= target_value:
            return i
    return None


def best_yield(campaign, target_name: str) -> float:
    vals = [r.metrics.get(target_name, float("-inf")) for r in campaign.results if r.success]
    return float(max(vals)) if vals else float("-inf")


def run(domain_name: str, optimizers: List[str], n_seeds: int, n_initial: int,
        batch: int, iterations: int, use_prior: bool, prior_fraction: float,
        out_path: str):
    proto = get_domain(domain_name)()
    target_name = proto.target
    target_value = proto.target_value
    budget = n_initial + iterations * batch
    ds_max = max(proto._table.values())

    print(f"\n  TIER-1 RETROSPECTIVE VALIDATION on REAL data: '{domain_name}'")
    print(f"  dataset reactions={len(proto._table)}  best measured yield={ds_max:.1f}  "
          f"target>={target_value}  budget={budget} experiments  seeds={n_seeds}")
    # Baseline from the data alone: picking measured reactions uniformly at
    # random without repeats, the expected number of picks until the first one
    # at or above target is (N + 1) / (K + 1).
    n_total = len(proto._table)
    n_at_target = sum(v >= target_value for v in proto._table.values())
    random_expected = (n_total + 1) / (n_at_target + 1)
    print(f"  reactions at or above target={n_at_target} ({100 * n_at_target / n_total:.2f}%)  "
          f"expected random picks to find one={random_expected:.0f}")
    if use_prior:
        print(f"  physics_bo warm-started from a historical prior "
              f"({int(prior_fraction*100)}% of the table)")
    print()
    header = f"  {'optimizer':<14}{'reached target':>16}{'median exps':>14}{'mean exps':>12}{'mean best%':>12}"
    print(header + "\n  " + "-" * (len(header) - 2))

    results = {}
    for opt_name in optimizers:
        etts, bests, reached = [], [], 0
        for seed in range(n_seeds):
            domain = get_domain(domain_name)()
            if use_prior and opt_name in ("physics_bo", "hybrid"):
                domain.enable_prior(fraction=prior_fraction, seed=1000 + seed)
            opt = get_optimizer(opt_name)(domain.parameter_space(), domain.objectives(), seed=seed)
            ex = TabularExecutor(domain, noise=0.0, seed=seed)
            cfg = CampaignConfig(n_iterations=iterations, batch_size=batch,
                                 n_initial=n_initial, seed=seed, target_stop=True)
            campaign = Orchestrator(domain, opt, ex, cfg).run()
            ett = experiments_to_target(campaign, target_name, target_value)
            bests.append(best_yield(campaign, target_name))
            if ett is not None:
                etts.append(ett)
                reached += 1
        rate = reached / n_seeds
        med = median(etts) if etts else None
        avg = mean(etts) if etts else None
        results[opt_name] = {
            "reach_rate": rate, "median_ett": med, "mean_ett": avg,
            "mean_best": mean(bests), "etts": etts, "n_seeds": n_seeds, "budget": budget,
        }
        med_s = f"{med:.0f}" if med is not None else "-"
        avg_s = f"{avg:.1f}" if avg is not None else "-"
        print(f"  {opt_name:<14}{rate*100:>14.0f}%{med_s:>14}{avg_s:>12}{mean(bests):>12.1f}")

    # Ratios of mean experiments-to-target. The means only cover seeds that
    # reached the target, so a ratio is only a fair comparison when both
    # optimizers reached it in every seed; otherwise report the reach rates.
    def _compare(name, r, ref_name):
        ref = results.get(ref_name)
        if not ref or not r["mean_ett"] or not ref["mean_ett"]:
            return
        if r["reach_rate"] == 1.0 and ref["reach_rate"] == 1.0:
            print(f"  {name}: {ref['mean_ett'] / r['mean_ett']:.1f}x fewer experiments than {ref_name}")
        else:
            print(f"  {name} vs {ref_name}: reached target in {r['reach_rate']*100:.0f}% vs "
                  f"{ref['reach_rate']*100:.0f}% of seeds (means cover successful seeds only)")

    print()
    for opt_name, r in results.items():
        if opt_name != "random":
            _compare(opt_name, r, "random")
        if opt_name not in ("random", "bayesian"):
            _compare(opt_name, r, "bayesian")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"domain": domain_name, "target_value": target_value,
                   "dataset_max": ds_max, "n_at_target": n_at_target,
                   "random_expected_ett": random_expected,
                   "use_prior": use_prior, "results": results}, fh, indent=2)
    print(f"\n  saved -> {out_path}\n")
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tier-1 retrospective validation on real data")
    ap.add_argument("--domain", default="buchwald")
    ap.add_argument("--optimizers", nargs="+", default=["random", "bayesian", "physics_bo"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--initial", type=int, default=10)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=38)
    ap.add_argument("--prior", action="store_true", help="warm-start physics_bo from a historical prior")
    ap.add_argument("--prior-fraction", type=float, default=0.30)
    ap.add_argument("--out", default="results/retro_buchwald.json")
    args = ap.parse_args(argv)
    run(args.domain, args.optimizers, args.seeds, args.initial, args.batch,
        args.iterations, args.prior, args.prior_fraction, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
