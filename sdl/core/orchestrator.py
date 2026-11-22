"""The generic closed-loop orchestrator.

propose -> execute -> observe -> update, repeated. Three phases (salvaged from
the perovskite pipeline but domain-free):
  1. exploration  - random space-filling to seed the surrogate,
  2. optimization - the chosen optimizer drives the loop,
  3. validation   - replicate the best candidate to estimate reproducibility.

Knows nothing about any field; it only manipulates Domain / Optimizer /
Executor / ObjectiveSet through their interfaces.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from sdl.core.domain import Domain
from sdl.core.executor import Executor
from sdl.core.experiment import Campaign, Experiment, ExperimentResult
from sdl.core.optimizer import Optimizer
from sdl.core.store import ExperimentStore
from sdl.optimizers.random_search import RandomSearch

logger = logging.getLogger("sdl.orchestrator")


@dataclass
class CampaignConfig:
    n_iterations: int = 20
    batch_size: int = 4
    n_initial: int = 8          # random exploration experiments before optimizing
    n_validation: int = 0       # replicates of the best candidate at the end
    target_stop: bool = True    # stop early once all objective targets are met
    seed: Optional[int] = None
    store_root: Optional[str] = None


class Orchestrator:
    def __init__(
        self,
        domain: Domain,
        optimizer: Optimizer,
        executor: Executor,
        config: Optional[CampaignConfig] = None,
    ):
        self.domain = domain
        self.optimizer = optimizer
        self.executor = executor
        self.config = config or CampaignConfig()
        self.objectives = domain.objectives()
        self.space = domain.parameter_space()
        self.rng = np.random.default_rng(self.config.seed)
        self._explorer = RandomSearch(self.space, self.objectives, seed=self.config.seed)
        # Hand physics-aware optimizers the domain's cheap prior once. No-op for
        # optimizers that don't override bind_domain.
        self.optimizer.bind_domain(domain)
        # Share the *same* ObjectiveSet the orchestrator scales each iteration, so
        # the optimizer's surrogate scalarizes with properly std-normalised
        # objectives instead of the saturating max(|s|,1.0) fallback. Without this
        # the multi-objective scalar collapses and BO degenerates toward random.
        self.optimizer.objectives = self.objectives
        self._explorer.objectives = self.objectives

    def run(self) -> Campaign:
        cfg = self.config
        campaign = Campaign(domain_name=self.domain.name, config=cfg.__dict__.copy())
        store = ExperimentStore(cfg.store_root, campaign.campaign_id) if cfg.store_root else None

        logger.info("Campaign %s on domain '%s' starting", campaign.campaign_id, self.domain.name)

        # Phase 1: exploration
        if cfg.n_initial > 0:
            seed_exps = self._explorer.propose(cfg.n_initial, [])
            self._execute_and_record(seed_exps, campaign, store, iteration=0)
            self.optimizer.observe(campaign.successful_results())

        # Phase 2: optimization
        for it in range(1, cfg.n_iterations + 1):
            campaign.iterations_completed = it
            self.objectives.update_scales([r.metrics for r in campaign.successful_results()])
            proposals = self.optimizer.propose(cfg.batch_size, campaign.successful_results())
            for p in proposals:
                p.iteration = it
            self._execute_and_record(proposals, campaign, store, iteration=it)
            self.optimizer.observe(campaign.successful_results())

            best = campaign.best_result
            logger.info(
                "iter %d/%d | experiments=%d | best_scalar=%.4f",
                it, cfg.n_iterations, campaign.n_experiments, campaign.best_scalar,
            )
            if cfg.target_stop and best and self.objectives.all_targets_met(best.metrics):
                logger.info("All objective targets met at iteration %d; stopping early.", it)
                break

        # Phase 3: validation (replicate the best candidate)
        if cfg.n_validation > 0 and campaign.best_result is not None:
            reps = [
                Experiment(
                    params=campaign.best_result.params,
                    proposed_by="validation",
                    iteration=campaign.iterations_completed,
                    metadata={"validation_replicate": i},
                )
                for i in range(cfg.n_validation)
            ]
            val_results = self._execute_and_record(reps, campaign, store, iteration=campaign.iterations_completed)
            self._attach_reproducibility(campaign, val_results)

        if store:
            store.save_campaign(campaign)
        logger.info("Campaign %s done. Best scalar=%.4f", campaign.campaign_id, campaign.best_scalar)
        return campaign

    def _execute_and_record(
        self,
        experiments: List[Experiment],
        campaign: Campaign,
        store: Optional[ExperimentStore],
        iteration: int,
    ) -> List[ExperimentResult]:
        results = self.executor.run_batch(experiments)
        for r in results:
            r.iteration = iteration
            scalar = self.objectives.scalarize(r.metrics) if r.success else float("-inf")
            campaign.record(r, scalar)
            if store:
                store.append(r)
        return results

    def _attach_reproducibility(self, campaign: Campaign, val_results: List[ExperimentResult]) -> None:
        ok = [r for r in val_results if r.success]
        if len(ok) < 2:
            return
        repro = {}
        for name in self.objectives.names:
            vals = [r.metrics.get(name) for r in ok if name in r.metrics]
            if len(vals) >= 2:
                repro[name] = float(np.std(vals))
        if campaign.best_result is not None:
            campaign.best_result.metadata["reproducibility_std"] = repro
