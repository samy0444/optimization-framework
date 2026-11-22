# Optimization Framework

A closed-loop optimization engine for problems where every experiment is
expensive: it proposes experiments, runs them, learns from the results and
proposes the next batch. The engine is domain-agnostic. A field plugs in by
describing its parameters, its objectives and, optionally, a cheap
physics-based prior, and the same optimizers then run on it unchanged.

The repository has two parts:

| Path | What it is |
|---|---|
| [`sdl/`](sdl) | The domain-agnostic engine: core loop, optimizers, domains, benchmarks, CLI and HTTP API |
| [`tests/`](tests) | Unit and integration tests for the engine |
| [`results/`](results) | Raw JSON from the benchmark runs reported below |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Benchmark setups, full results and caveats |
| [`applications/perovskite/`](applications/perovskite) | An earlier perovskite solar-cell application: Bayesian optimization for composition plus a reinforcement-learning agent for fabrication settings |

## How the engine works

```
            ┌──────────────── Domain ────────────────┐
            │ parameter space · objectives · prior   │
            └────────────────────────────────────────┘
                                │
   propose (Optimizer) ──► run (Executor) ──► observe ──► update ──┐
          ▲                                                       │
          └───────────────────────────────────────────────────────┘
```

- **Domain** ([`sdl/core/domain.py`](sdl/core/domain.py)): declares the
  parameter space (continuous and categorical), the objectives and an optional
  `simulate()` prior, which is a cheap physics or analytic model of the outcome.
- **Objectives** ([`sdl/core/objective.py`](sdl/core/objective.py)): each one has
  a direction, a weight, an optional target and an optional hard bound. Several
  objectives are combined by weighted scalarization, and a Pareto-front helper
  is included.
- **Orchestrator** ([`sdl/core/orchestrator.py`](sdl/core/orchestrator.py)):
  space-filling exploration, then optimizer-driven iterations, then replicates
  of the best candidate. It stops early once every objective target is met.
- **Executors**: `simulation` evaluates a domain's ground truth with optional
  noise. `tabular` looks up real measured outcomes from a dataset and reports a
  failed experiment for anything not in the table, so it never invents a value.

### Optimizers

| Name | Method |
|---|---|
| `random` | Uniform random sampling (baseline) |
| `bayesian` | Gaussian-process regression (scikit-learn) with Expected Improvement, Upper Confidence Bound or Thompson sampling, maximised over a large candidate pool |
| `physics_bo` | Bayesian optimization on the residual between observations and the domain's prior: the model is `prior(x) + GP_residual(x)`, so data is spent only on what the prior gets wrong ([`sdl/optimizers/physics_bo.py`](sdl/optimizers/physics_bo.py)) |
| `rl` | Soft Actor-Critic (stable-baselines3) trained on a surrogate fit to the campaign history, so policy training uses no real experiments |
| `hybrid` | Splits each batch between Bayesian optimization and the SAC policy, shifting towards RL as data accumulates |

`rl` and `hybrid` fall back to random proposals if stable-baselines3 and
gymnasium are not installed.

### Domains

| Name | Type | Objectives |
|---|---|---|
| `buchwald` | **Real data**: 3,955 measured Buchwald–Hartwig reactions (4 categorical factors) | yield |
| `reaction` | Simulated reaction: an Arrhenius-style prior plus a deliberate "reality gap" in the ground truth | yield, purity |
| `perovskite` | Simulated solar cell: a physics prior for bandgap, Jsc, Voc, fill factor and a stability proxy | efficiency, T80 |
| `branin` | Branin test function as the prior; the ground truth adds a bias well, so the true optimum is misranked by the prior | branin value |

Adding a field means writing one `Domain` subclass. [`sdl/domains/branin.py`](sdl/domains/branin.py)
is a complete example in about 60 lines.

## Results

All results come from simulation or from replaying a real published dataset.
None is a live lab result. Full setups and per-optimizer numbers are in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

**Real data: replaying 3,955 measured Buchwald–Hartwig reactions** (20 seeds,
target ≥ 95% yield, budget 200 experiments):

| Optimizer | Seeds reaching target | Mean experiments to target |
|---|---:|---:|
| `random` | 60% | 67.3 |
| `bayesian` | 100% | 29.0 |
| `physics_bo` | 100% | 29.0 |

Bayesian optimization reaches the target in every seed with 2.3× fewer
experiments than random search. `physics_bo` ties standard BO here because the
dataset is categorical, so there is no physics prior to use.

**Simulation: a reaction whose ground truth deviates from its kinetic prior**
(5 seeds, budget 51 experiments): `physics_bo` meets both the yield (≥ 90%) and
purity (≥ 99%) targets in 3 of 5 seeds, in as few as 9 experiments. Standard
Bayesian optimization and random search meet them in none.

## Quick start

Python 3.8 or later.

```bash
pip install -r requirements.txt
python -m pytest -q tests/

python -m sdl.cli list                                    # registered domains, optimizers, executors
python -m sdl.cli run --domain branin --optimizer bayesian --iterations 20
python -m sdl.cli run --domain perovskite --optimizer hybrid --iterations 10
python -m sdl.demo                                        # compare optimizers on one domain

uvicorn sdl.api.app:app --port 8000                       # HTTP API; docs at http://localhost:8000/docs
```

Reproduce the benchmarks:

```bash
python -m sdl.benchmarks.retrospective --seeds 20 --out results/retro_buchwald.json
python -m sdl.benchmarks.runner --domain reaction --optimizers random bayesian physics_bo \
    --seeds 0 1 2 3 4 --iterations 15 --batch 3 --initial 6 --noise 0.01 \
    --out results/bench_reaction.json
```

## Perovskite application

[`applications/perovskite/`](applications/perovskite) is the project this engine
grew out of: a two-level loop in which a Gaussian process (BoTorch) proposes
Cs/FA/MA compositions from MIT PV-Lab degradation data, a Soft Actor-Critic
agent tunes six fabrication settings for each composition, and neural-network
surrogates score the results. It also includes a DeepXDE physics-informed
neural network used to generate synthetic lifetime data. See its
[README](applications/perovskite/README.md) for setup, surrogate accuracy and
known limitations.

## Data

- **Buchwald–Hartwig dataset**, included as
  [`sdl/data/buchwald_hartwig.csv`](sdl/data/buchwald_hartwig.csv): Ahneman,
  Estrada, Lin, Dreher and Doyle, "Predicting reaction performance in C–N
  cross-coupling using machine learning," *Science* 360, 186–190 (2018). The
  authors publish the data in [doylelab/rxnpredict](https://github.com/doylelab/rxnpredict)
  under the MIT License.
- Perovskite datasets are linked as git submodules; see the
  [application README](applications/perovskite/README.md#data).

## License

The code is released under the [MIT License](LICENSE). Datasets remain under
the terms of their original sources.
