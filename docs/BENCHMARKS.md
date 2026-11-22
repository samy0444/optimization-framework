# Benchmarks

Every number below comes from the JSON files in [`results/`](../results) and
can be regenerated with the command shown. Nothing here is a live lab result:
the benchmarks are either simulations or replays of a published dataset.

## 1. Buchwald–Hartwig: replay of real measured data

**Data.** 3,955 Buchwald–Hartwig amination reactions from a high-throughput
screen (Ahneman et al., *Science*, 2018), with four categorical factors (ligand,
additive, base, aryl halide) and measured yield from 0 to 100%.

**Protocol.** The optimizer chooses which reactions to "run". The `tabular`
executor returns only the real measured yield of each pick, and any combination
not in the table counts as a failed experiment, so no value is ever simulated.
The target is a reaction with at least 95% yield.

**How hard the target is.** Only 19 of the 3,955 reactions (0.48%) reach 95%
yield; the mean yield is 33%. Picking measured reactions uniformly at random
without repeats, the expected number of picks until the first such reaction is
(N + 1) / (K + 1) = 3956 / 20 ≈ 198, and the chance of finding one within 200
picks is about 63%.

**Setup.** 20 seeds; budget of 200 experiments (10 initial, then 38 rounds of 5).

| Optimizer | Seeds reaching ≥95% | Median experiments | Mean experiments* | Mean best yield |
|---|---:|---:|---:|---:|
| `random` | 60% (12 of 20) | 57.5 | 67.3 | 96.1% |
| `bayesian` | 100% (20 of 20) | 20.5 | 29.0 | 98.6% |
| `physics_bo` | 100% (20 of 20) | 20.5 | 29.0 | 98.6% |

\*Median and mean are taken over the seeds that reached the target.

Both model-based optimizers reach the target in every seed, after 29
experiments on average (worst seed: 108), about 6.8× fewer than the 198 expected
for random picking. Random search reaches it in 12 of the 20 seeds, in line with
the 63% predicted above.
Those 12 seeds need 67.3 experiments on average, and the other 8 use the whole
200-experiment budget without reaching it. Because the random-search average
leaves out its failed seeds, it understates the gap.

`physics_bo` and `bayesian` tie exactly here. The dataset is purely
categorical, so there is no physics model to supply a prior, and with the prior
switched off `physics_bo` reduces to standard Bayesian optimization.

```bash
python -m sdl.benchmarks.retrospective --seeds 20 --out results/retro_buchwald.json
```

### Warm start from historical data (`--prior`)

`physics_bo` can be warm-started from a model fit on a random 30% of the same
table, to show how earlier experiments can speed up a new campaign.

No numbers are reported for this mode. The historical 30% is drawn from the
same table that is then optimized, so any speed-up would overstate what a new
campaign could expect. A fair test would fit the prior on one region of
chemical space and optimize on a disjoint one; that is not implemented yet.

```bash
python -m sdl.benchmarks.retrospective --seeds 20 --prior --out results/retro_buchwald_prior.json
```

## 2. Simulated reaction with a reality gap

**Domain.** Six process conditions (temperature, time, catalyst loading,
equivalents, concentration, solvent) with two objectives: yield (target ≥ 90%)
and purity (target ≥ 99%). The optimizer's prior is an Arrhenius-style kinetic
model; the ground truth adds effects the prior misses, such as catalyst
deactivation at high temperature and solvent-dependent impurities.

**Setup.** 5 seeds; 6 initial experiments, then up to 15 rounds of 3 (51
experiments); measurement noise 0.01. A run stops early once both targets are met.

| Optimizer | Seeds meeting both targets | Experiments used (seeds 0–4) | Best yield across seeds | Best purity across seeds |
|---|---:|---|---|---|
| `physics_bo` | **3 of 5** | 9, 36, 51, 24, 51 | 87.1–96.2% | 98.0–99.2% |
| `bayesian` | 0 of 5 | 51 in every seed | 62.5–92.6% | 78.0–95.3% |
| `random` | 0 of 5 | 51 in every seed | 70.5–82.6% | 84.3–97.9% |

Starting from the prior and modelling only its error lets `physics_bo` meet
both targets in three seeds, in as few as 9 experiments. Standard Bayesian
optimization, which starts from no prior, and random search do not meet them
in any seed within the budget.

```bash
python -m sdl.benchmarks.runner --domain reaction --optimizers random bayesian physics_bo \
    --seeds 0 1 2 3 4 --iterations 15 --batch 3 --initial 6 --noise 0.01 \
    --out results/bench_reaction.json
```

## Limitations

- Section 2 is a simulation. It shows the method is sample-efficient when the
  prior is informative but imperfect; it does not show a lab outcome.
- Section 1 uses real measurements but replays a fixed, fully enumerated
  table. A live campaign adds noise, failed runs and options outside the table.
- Seed counts are small (5 and 20), so differences between close results should
  not be over-read.
