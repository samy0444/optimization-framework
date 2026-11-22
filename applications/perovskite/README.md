# Perovskite application: BO + RL closed loop

A perovskite solar-cell design loop built before the domain-agnostic engine in
[`sdl/`](../../sdl). It splits the problem into two levels:

1. **Composition (Bayesian optimization).** A BoTorch Gaussian process with
   q-Expected Improvement proposes Cs/FA/MA compositions that sum to 1
   ([`rl_bo_pipeline/physics_informed_bo.py`](rl_bo_pipeline/physics_informed_bo.py)).
   It starts from the MIT PV-Lab degradation data: each sample's instability
   is its colour change over time (from calibrated RGB images), doubled when
   its XRD spectrum shows impurity peaks, and the GP maximizes stability.
   Once the loop has three results of its own, the GP is refit on those
   instead, so one model never mixes two different quantities.
2. **Fabrication (reinforcement learning).** For each proposed composition, a
   Soft Actor-Critic agent (stable-baselines3) is fine-tuned from a base agent
   to choose six process settings: substrate, Pb fraction, UV filter, spin
   speed, annealing temperature and encapsulation
   ([`rl_bo_pipeline/environment_rl_bo.py`](rl_bo_pipeline/environment_rl_bo.py)).
3. **Evaluation.** Neural-network surrogates score each design for efficiency
   and log-lifetime ([`surrogate_model.py`](surrogate_model.py)). The product
   efficiency × log-lifetime is fed back to the BO for the next cycle
   ([`rl_bo_pipeline/tri_loop.py`](rl_bo_pipeline/tri_loop.py)).

All runs use the same five-layer device stack, defined once in
[`layers.py`](layers.py) (`default_stack()`), with values inside the range the
lifetime surrogate was trained on.

## Setup

Run everything from this folder. File paths in the code are relative to it.

```bash
git submodule update --init applications/perovskite/cluster-perovskite-data   # from the repo root
cd applications/perovskite
pip install -r requirements.txt
```

`perovskitedatabase_data` is only needed to retrain the surrogates
(`train_hybrid_surrogate.py`); initialise it the same way if you need it.

## Run

```bash
streamlit run app.py                  # UI: manual simulation, BO+RL loop, and the
                                      # random / BO-only / RL-only / BO+RL benchmark
python evaluate_hybrid_surrogate.py   # surrogate accuracy on the held-out test sets
python train_base_agent_rl_bo.py      # retrain the base SAC agent (writes fab_agent.zip)
```

A trained base agent (`fab_agent.zip`) and trained surrogates are included, so
the loop runs without retraining. The benchmark's RL-only arm trains its own
agents and does not overwrite `fab_agent.zip`. All four benchmark arms score
designs with the same simulator settings.

## Surrogates

Two MLPs, trained by [`train_hybrid_surrogate.py`](train_hybrid_surrogate.py):

| Surrogate | Training data | Held-out test set | RMSE | R² |
|---|---|---|---|---|
| Efficiency (32 inputs) | synthetic device data + Perovskite Database entries | 17,791 samples | 3.73 | 0.712 |
| Log-lifetime (30 inputs) | synthetic encapsulation data + 28 MIT PV-Lab samples | 3,926 samples | 0.108 | 0.985 |

Test sets are random 20% splits of the same mixed synthetic and real pools, so
these numbers mostly measure fit to the synthetic generators. The lifetime test
set has only a handful of real samples.

The synthetic lifetime data comes from
[`synthetic_datagen/lifetime_data_gen.py`](synthetic_datagen/lifetime_data_gen.py),
which uses a DeepXDE physics-informed neural network
([`synthetic_datagen/train_pinn_model.py`](synthetic_datagen/train_pinn_model.py))
for a 2-D diffusion model of degradation.

## Known limitations

- **The efficiency surrogate's training data is inconsistent.** Its synthetic
  part is a simple formula of composition and process settings scaled 0–1,
  while the Perovskite Database rows give efficiency in percent (some parsed
  values reach 168). The synthetic layer features also use different scales
  from the lifetime data. Efficiency predictions are therefore rough;
  regenerating the data on one scale and retraining is open work.
- **Lifetime depends only on the layer stack.** Composition and process
  settings change the score only through efficiency and a bandgap correction.
- Everything here is simulation. No result in this folder is a lab measurement.

## Data

- [`cluster-perovskite-data`](https://github.com/PV-Lab/cluster-perovskite-data):
  MIT PV-Lab perovskite degradation data (git submodule, MIT license).
- [`perovskitedatabase_data`](https://github.com/Jesperkemist/perovskitedatabase_data):
  the Perovskite Database Project (Jacobsson et al., *Nature Energy*, 2022)
  (git submodule). See the source for its terms of use.
