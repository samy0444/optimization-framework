# optimization-framework

This project is about a simple question: **if every experiment is slow and
expensive, how do you decide which ones to run?**

Think of a chemist trying to get the best yield out of a reaction. They can
change the temperature, the reaction time, the amount of catalyst, the solvent
and so on. Each attempt might take hours, and there are far too many
combinations to try them all. People usually rely on experience, a few rules of
thumb and some trial and error.

The code in this repository does that choosing automatically. It looks at the
results so far, works out which experiments are most worth trying next, asks
for them to be run, learns from the new results, and repeats. This is often
called closed-loop optimization, and when it is connected to automated lab
equipment people call the whole setup a "self-driving lab". Here the
experiments are either simulations or replays of real published data; nothing
is connected to physical lab equipment.

The code is written so that the same optimization methods can be reused on
completely different problems. So far I have used it on chemical reactions and
on perovskite solar cells.

## How it works, step by step

1. **Start by spreading out.** The first few experiments are chosen to cover
   the range of possible settings, so the method gets a rough picture of the
   whole space.
2. **Build a model of what has been seen.** From the results so far, the
   method builds a model that predicts the outcome for settings it has not
   tried yet, and also says how sure it is about each prediction.
3. **Choose the next experiments.** It picks settings where the model expects
   a good result, or where it is very unsure and a test would teach it a lot.
   Balancing those two is the core of the method.
4. **Run them and update.** The new results are added and the model is
   rebuilt.
5. **Stop** when the goal is reached or the experiment budget runs out. At the
   end it can repeat the best experiment a few times to check that the result
   holds up.

The way to judge a method like this is simple: how many experiments does it
need before it finds a good answer? Fewer is better, because in a real lab
every experiment costs time and money.

## Starting from what we already know

In many fields there is already a rough formula for how the system behaves,
for example textbook reaction kinetics. That formula is usually close to
reality but not exactly right.

The `physics_bo` method uses this. Instead of learning everything from scratch,
it starts from the formula and spends its experiments learning only where the
formula is wrong. When the formula is a decent guess, this can save a lot of
experiments. When there is no formula at all, it behaves exactly like ordinary
Bayesian optimization.

## The optimization methods

| Name | What it does |
|---|---|
| `random` | Picks settings at random. It is the baseline every other method should beat. |
| `bayesian` | Bayesian optimization. It fits a Gaussian process (a model that gives a prediction together with an uncertainty) and chooses experiments using Expected Improvement, Upper Confidence Bound or Thompson sampling, three standard ways of trading off "likely good" against "worth exploring". |
| `physics_bo` | Bayesian optimization that starts from the problem's physics formula and learns only the difference between that formula and the observed results. |
| `rl` | Reinforcement learning with Soft Actor-Critic (from stable-baselines3). The agent practises on a model fitted to past results, so it does not use up real experiments while it learns. |
| `hybrid` | Splits each batch of experiments between Bayesian optimization and the reinforcement-learning agent, giving more of the batch to the agent as more data comes in. |

Some problems have more than one goal, for example high yield *and* high
purity. These goals are combined into one score using weights. Each goal can
also have a target ("90% is good enough") and a hard limit ("never below
zero").

## How the code is organised

The code is split into a few pieces that each do one job, so a new problem can
be added without touching the optimization methods.

- **A problem** (called a `Domain` in the code) describes the settings that can
  be changed, what counts as a good result, and, if one exists, a cheap formula
  that roughly predicts the outcome.
- **An optimizer** decides which experiments to run next. It never sees what
  the problem is about, only numbers.
- **An executor** "runs" the experiments. For simulated problems it calculates
  the true outcome. For real datasets it looks up the measured result, and if
  the requested combination was never measured it reports a failed experiment
  instead of inventing a number.
- **The loop** (`sdl/core/orchestrator.py`) ties these together and repeats the
  steps above.

### Folders

- `sdl/`: the optimization code
  - `core/`: the loop and the basic building blocks
  - `optimizers/`: the methods listed above
  - `domains/`: the example problems
  - `benchmarks/`: scripts that compare methods
  - `cli.py` and `api/`: a command-line tool and a small web API
- `tests/`: automated tests
- `results/`: the raw output of the benchmark runs described below
- `docs/BENCHMARKS.md`: full details of the benchmarks
- `applications/perovskite/`: an earlier, separate solar-cell project (see below)

## The example problems

| Name | What it is |
|---|---|
| `buchwald` | **Real data.** 3,955 Buchwald–Hartwig coupling reactions that were all actually run and measured in a high-throughput experiment (Ahneman et al., *Science*, 2018). The settings are the choice of ligand, additive, base and aryl halide; the result is the yield. |
| `reaction` | A simulated reaction with six settings (temperature, time, catalyst amount, reagent ratio, concentration and solvent) and two goals, yield and purity. The "true" behaviour deliberately differs from the textbook formula, the way real chemistry does. |
| `perovskite` | A simulated perovskite solar cell, using standard physics relations for the bandgap, voltage, current and a stability estimate. |
| `branin` | A well-known mathematical test function, included as a simple check. |

## Results

### Real data: Buchwald–Hartwig reactions

Finding a good reaction here is hard. Only 19 of the 3,955 reactions (about
0.5%) reach 95% yield, and the average yield is 33%. If you picked reactions at
random, never repeating one, you would expect to test about 198 before hitting
one of those 19.

In this dataset every combination has already been measured, which makes a
fair test possible. The method is not shown the measured yields. It chooses
which reactions it would like to "run", and only then is it told the real
measured yield of those reactions. The question is how many reactions it needs
before it finds one with at least 95% yield. Each method was run 20 times with
different random starts, with a limit of 200 experiments per run.

| Method | Runs that found a ≥95% reaction | Average experiments needed, in the runs that found one |
|---|---|---|
| random | 12 of 20 | 67.3 |
| bayesian | 20 of 20 | 29.0 |
| physics_bo | 20 of 20 | 29.0 |

Bayesian optimization found a high-yield reaction in all 20 runs, after 29
experiments on average. That is less than 1% of the reactions, and about 7
times fewer than the 198 that random picking needs on average. Even its slowest
run needed 108.

The random runs in the table behave as that calculation predicts. They found
a high-yield reaction in only 12 of the 20 runs. In those 12 they needed 67
experiments on average, and in the other 8 they used up all 200 experiments
without finding one. So the 67 only counts the runs where random picking got
lucky.

`physics_bo` gives exactly the same result as `bayesian` here. That is
expected: this dataset only has categories (which ligand, which base), so
there is no physics formula to start from, and without one the two methods are
identical.

### Simulation: a reaction that doesn't follow the textbook

Here the method knows a textbook formula for the reaction, but the simulated
"real" reaction behaves differently in places. Each method got up to 51
experiments, and the goal was at least 90% yield *and* at least 99% purity.
Each method was run 5 times.

`physics_bo` reached both goals in 3 of the 5 runs, in one case after only 9
experiments. Ordinary Bayesian optimization and random picking did not reach
them in any run. This is where starting from an imperfect formula helps.

### What these results do and don't show

They show that the methods use fewer experiments than random search, both on
real measured chemistry and in simulation. They do not show a result from a
real lab: no experiment here was run live. The simulated problems are
realistic in shape, but their exact numbers are made up. See
`docs/BENCHMARKS.md` for the full setup.

## Getting started

You need Python 3.8 or later. Install the packages and run the tests:

```bash
pip install -r requirements.txt
python -m pytest -q tests/
```

Then try it from the command line:

```bash
python -m sdl.cli list                       # shows the available problems and methods
python -m sdl.cli run --domain branin --optimizer bayesian --iterations 20
python -m sdl.cli run --domain reaction --optimizer physics_bo --iterations 15
python -m sdl.demo                           # compares several methods on one problem
```

Each `run` prints a summary at the end: how many experiments were used and the
best settings and result found.

To reproduce the two benchmarks above (the first takes a few minutes, the
second about a minute on a laptop):

```bash
python -m sdl.benchmarks.retrospective --seeds 20 --out results/retro_buchwald.json
python -m sdl.benchmarks.runner --domain reaction --optimizers random bayesian physics_bo \
    --seeds 0 1 2 3 4 --iterations 15 --batch 3 --initial 6 --noise 0.01 \
    --out results/bench_reaction.json
```

There is also a small web API if you want to run campaigns from another
program: start it with `uvicorn sdl.api.app:app --port 8000` and open
`http://localhost:8000/docs` to see what it offers.

## Adding your own problem

A new problem is a single Python class. You describe the settings, the goal,
and optionally a rough formula. Here is a made-up example about annealing a
thin film, where the textbook guess is fine except that films made with DMSO
degrade above 160 °C:

```python
import math

from sdl.core.domain import Domain
from sdl.core.objective import Direction, Objective, ObjectiveSet
from sdl.core.orchestrator import CampaignConfig, Orchestrator
from sdl.core.registry import get_executor, get_optimizer, register_domain
from sdl.core.space import CategoricalParameter, ContinuousParameter, ParameterSpace


@register_domain("film_anneal")
class FilmAnneal(Domain):
    """Toy example: choose annealing conditions for a thin film."""

    def parameter_space(self):
        return ParameterSpace(parameters=[
            ContinuousParameter("temperature_C", 80.0, 200.0),
            ContinuousParameter("time_min", 1.0, 30.0),
            CategoricalParameter("solvent", ["DMF", "DMSO", "GBL"]),
        ])

    def objectives(self):
        return ObjectiveSet([Objective("quality", Direction.MAXIMIZE, target=90.0)])

    def simulate(self, p):
        # The "textbook" guess: best near 150 C, and longer is better.
        peak = math.exp(-((p["temperature_C"] - 150.0) / 40.0) ** 2)
        return {"quality": 100.0 * peak * (1.0 - math.exp(-p["time_min"] / 8.0))}

    def ground_truth(self, p):
        # "Reality": the same, except DMSO films degrade above 160 C.
        quality = self.simulate(p)["quality"]
        if p["solvent"] == "DMSO" and p["temperature_C"] > 160.0:
            quality *= 0.6
        return {"quality": quality}


domain = FilmAnneal()
optimizer = get_optimizer("physics_bo")(domain.parameter_space(), domain.objectives(), seed=0)
executor = get_executor("simulation")(domain, noise=0.01, seed=0)
config = CampaignConfig(n_iterations=10, batch_size=3, n_initial=6, seed=0)

campaign = Orchestrator(domain, optimizer, executor, config).run()
print(campaign.summary())
```

Run it from the repository's top folder. In a test run it found a film quality
of about 92, above the target of 90, after 9 experiments and then stopped on
its own. If you have no formula, leave out `simulate()` and use the `bayesian`
optimizer instead. `sdl/domains/branin.py` is another short example.

## The perovskite solar-cell project

`applications/perovskite/` is the project this framework grew out of. It
tackles perovskite solar cells in two layers:

- a Bayesian optimizer chooses the material composition (the mix of caesium,
  formamidinium and methylammonium), starting from real degradation
  measurements published by MIT's PV-Lab;
- for each composition, a reinforcement-learning agent chooses the fabrication
  settings, such as spin speed, annealing temperature and encapsulation;
- neural-network models trained on simulated and published data predict the
  efficiency and lifetime of each design. One part of the simulated lifetime
  data comes from a physics-informed neural network.

It has its own README with setup steps, model accuracy and an honest list of
known limitations.

## Data and license

The Buchwald–Hartwig data (`sdl/data/buchwald_hartwig.csv`) comes from Ahneman,
Estrada, Lin, Dreher and Doyle, "Predicting reaction performance in C–N
cross-coupling using machine learning," *Science* 360, 186–190 (2018). The
authors share it under the MIT License in
[doylelab/rxnpredict](https://github.com/doylelab/rxnpredict). The perovskite
datasets are linked as git submodules and keep their own terms.

The code in this repository is released under the [MIT License](LICENSE).
