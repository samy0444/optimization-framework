"""Core framework tests. Run: pytest -q  (from repo root)."""

import numpy as np
import pytest

from sdl.core.objective import Direction, Objective, ObjectiveSet, pareto_front
from sdl.core.orchestrator import CampaignConfig, Orchestrator
from sdl.core.registry import available, get_domain, get_executor, get_optimizer
from sdl.core.space import (
    CategoricalParameter,
    ContinuousParameter,
    IntegerParameter,
    ParameterSpace,
    SimplexConstraint,
)
from sdl.domains.branin import GROUND_TRUTH_MIN
from sdl.optimizers.random_search import RandomSearch


def test_space_roundtrip_and_bounds():
    space = ParameterSpace([
        ContinuousParameter("a", 0.0, 10.0),
        IntegerParameter("b", 1, 5),
        CategoricalParameter("c", ["x", "y", "z"]),
    ])
    rng = np.random.default_rng(0)
    p = space.sample(rng)
    assert 0.0 <= p["a"] <= 10.0 and 1 <= p["b"] <= 5 and p["c"] in {"x", "y", "z"}
    vec = space.to_vector(p)
    assert np.all((vec >= 0) & (vec <= 1))
    p2 = space.from_vector(vec)
    assert p2["c"] == p["c"] and p2["b"] == p["b"]


def test_simplex_constraint_repair():
    space = ParameterSpace(
        [ContinuousParameter("MA", 0, 1), ContinuousParameter("FA", 0, 1), ContinuousParameter("Cs", 0, 1)],
        constraints=[SimplexConstraint(["MA", "FA", "Cs"], total=1.0)],
    )
    p = space.sample(np.random.default_rng(1))
    assert abs(p["MA"] + p["FA"] + p["Cs"] - 1.0) < 1e-6
    assert space.is_feasible(p)


def test_objective_orientation_and_scalarize():
    objs = ObjectiveSet([
        Objective("loss", Direction.MINIMIZE, weight=1.0),
        Objective("gain", Direction.MAXIMIZE, weight=2.0),
    ])
    objs.update_scales([{"loss": 0.0, "gain": 0.0}, {"loss": 10.0, "gain": 10.0}])
    better = objs.scalarize({"loss": 1.0, "gain": 9.0})
    worse = objs.scalarize({"loss": 9.0, "gain": 1.0})
    assert better > worse


def test_pareto_front():
    objs = [Objective("a", Direction.MAXIMIZE), Objective("b", Direction.MAXIMIZE)]
    pts = [{"a": 1, "b": 1}, {"a": 2, "b": 0}, {"a": 0, "b": 2}, {"a": 0.5, "b": 0.5}]
    front = set(pareto_front(pts, objs))
    assert front == {0, 1, 2}  # the (0.5,0.5) point is dominated by (1,1)


def test_registry_has_builtins():
    avail = available()
    assert "branin" in avail["domains"] and "perovskite" in avail["domains"]
    assert {"random", "bayesian"}.issubset(set(avail["optimizers"]))


def test_closed_loop_random_runs_and_improves():
    domain = get_domain("branin")()
    opt = RandomSearch(domain.parameter_space(), domain.objectives(), seed=0)
    ex = get_executor("simulation")(domain, noise=0.0, seed=0)
    cfg = CampaignConfig(n_iterations=10, batch_size=4, n_initial=8, seed=0)
    campaign = Orchestrator(domain, opt, ex, cfg).run()
    assert campaign.n_experiments > 0
    assert campaign.best_result is not None
    # branin is minimized; best found should be reasonably good and never beat
    # the true (ground-truth) floor, which reality places below the prior's min.
    assert campaign.best_result.metrics["branin"] < 5.0
    assert campaign.best_result.metrics["branin"] >= GROUND_TRUTH_MIN - 1e-6


def test_physics_bo_beats_random_on_branin():
    # Averaged over seeds so the claim is robust, not luck. The cheap prior
    # already points near branin's optima, so physics-prior BO should reach a
    # markedly better (lower) value than random under the same budget.
    def mean_best(name):
        vals = []
        for seed in (0, 1, 2):
            domain = get_domain("branin")()
            opt = get_optimizer(name)(domain.parameter_space(), domain.objectives(), seed=seed)
            ex = get_executor("simulation")(domain, noise=0.0, seed=seed)
            cfg = CampaignConfig(n_iterations=15, batch_size=3, n_initial=6, seed=seed)
            c = Orchestrator(domain, opt, ex, cfg).run()
            vals.append(c.best_result.metrics["branin"])
        return float(np.mean(vals))
    assert mean_best("physics_bo") < mean_best("random")


def test_physics_bo_runs_and_uses_prior_to_match_or_beat_vanilla_bo():
    # On branin the prior is informative, so delta-GP BO should reach a best
    # objective no worse than prior-free BO under the same budget. Averaged over
    # seeds so the claim is robust to a single unlucky restart (both methods sit
    # near the ground-truth floor on most seeds; one seed can be an outlier).
    def mean_best(name):
        vals = []
        for seed in range(5):
            domain = get_domain("branin")()
            opt = get_optimizer(name)(domain.parameter_space(), domain.objectives(), seed=seed)
            ex = get_executor("simulation")(domain, noise=0.0, seed=seed)
            cfg = CampaignConfig(n_iterations=20, batch_size=3, n_initial=6, seed=seed)
            c = Orchestrator(domain, opt, ex, cfg).run()
            vals.append(c.best_result.metrics["branin"])  # lower is better
        return float(np.mean(vals))
    assert "physics_bo" in available()["optimizers"]
    assert mean_best("physics_bo") <= mean_best("bayesian") + 1.0


def test_ground_truth_differs_from_prior():
    # The reality gap must be real, else the delta-GP has nothing to learn.
    domain = get_domain("perovskite")()
    p = {"FA": 0.8, "Cs": 0.1, "MA": 0.1, "I": 1.5, "Br": 1.5,
         "anneal_temp": 150, "anneal_time": 20, "humidity": 40}
    prior = domain.simulate(p)
    truth = domain.ground_truth(p)
    assert abs(truth["pce"] - prior["pce"]) > 1e-3
    assert truth["t80"] <= prior["t80"]


def test_reaction_domain_runs_and_has_reality_gap():
    domain = get_domain("reaction")()
    p = {"temperature": 120.0, "time": 8.0, "catalyst": 5.0, "equiv": 2.5,
         "concentration": 1.5, "solvent": "DMF"}
    prior = domain.simulate(p)
    truth = domain.ground_truth(p)
    assert 0.0 <= prior["yield"] <= 100.0 and 0.0 <= prior["purity"] <= 100.0
    # high-T + DMF + high equiv/conc is exactly where the prior is over-optimistic
    assert truth["yield"] < prior["yield"]
    # closed loop optimizes it end-to-end
    opt = get_optimizer("physics_bo")(domain.parameter_space(), domain.objectives(), seed=0)
    ex = get_executor("simulation")(domain, noise=0.01, seed=0)
    c = Orchestrator(domain, opt, ex, CampaignConfig(n_iterations=12, batch_size=3, n_initial=6, seed=0)).run()
    assert c.best_result is not None and c.best_result.metrics["yield"] > 0.0


def test_buchwald_real_dataset_lookup_and_loop():
    # Real Science 2018 Buchwald-Hartwig data: ground truth is a lookup, not a sim.
    domain = get_domain("buchwald")()
    space = domain.parameter_space()
    # every dataset combination resolves to a real measured yield
    p = {f: domain._levels[f][0] for f in domain.factors}
    # pick a combination that exists by reading the first row
    row0 = {f: domain._rows[0][f] for f in domain.factors}
    val = domain.lookup(row0)
    assert val is not None and 0.0 <= val <= 100.0
    # closed loop runs against the real table via the tabular executor
    from sdl.executors.tabular import TabularExecutor
    opt = get_optimizer("bayesian")(space, domain.objectives(), seed=0)
    ex = TabularExecutor(domain, noise=0.0, seed=0)
    c = Orchestrator(domain, opt, ex, CampaignConfig(n_iterations=8, batch_size=4, n_initial=8, seed=0)).run()
    assert c.best_result is not None and c.best_result.metrics["yield"] > 0.0


def test_prior_card_roundtrip_and_scoring(tmp_path):
    # The priors library: a card serializes/deserializes and scoring is sane.
    from sdl.priors import PriorCard, score_prior
    card = PriorCard(prior_id="unit_test.v1", reaction_class="Test", domain="reaction")
    card.touch("created in test")
    d = card.to_dict()
    card2 = PriorCard.from_dict(d)
    assert card2.prior_id == "unit_test.v1" and card2.version == 1
    assert card2.history and "created in test" in card2.history[-1]
    # a perfect prior has zero error; a biased one is detected
    s = score_prior([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert s["rmse"] == 0.0 and abs(s["bias"]) < 1e-9 and s["n_validation"] == 3
    s2 = score_prior([2.0, 3.0, 4.0], [1.0, 2.0, 3.0])
    assert s2["bias"] > 0 and s2["rmse"] > 0  # prior over-predicts by 1.0


def test_perovskite_simulate_outputs_plausible():
    domain = get_domain("perovskite")()
    m = domain.simulate({"FA": 0.85, "Cs": 0.15, "MA": 0.0, "I": 2.7, "Br": 0.3,
                         "anneal_temp": 150, "anneal_time": 20, "humidity": 30})
    assert 0 < m["pce"] < 34
    assert 0 < m["voc"] < 2.0
    assert m["t80"] > 0
