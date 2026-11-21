# Autonomous Optimization Framework: Self-Driving Solar Lab for Perovskite Solar Cells

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository contains an advanced, autonomous optimization platform intended for perovskite solar cell production. It combines state-of-the-art methods like physics-informed surrogate modeling, Bayesian optimization (BO), reinforcement learning (RL), and generative AI to mimic a self-driving laboratory. The platform automatically finds best, stable, and high-efficiency solar cell compositions and production processes, greatly speeding up discovery and development.

---

## Repository Structure

```
├── app.py                      # Streamlit UI for interactive optimization and benchmarking
├── layers.py                   # Solar cell layer properties and configurations
├── interfaces.py               # Recombination parameters for the interface
├── simulation.py               # Physics-informed surrogate simulation model
├── benchmark_rl_bo.py          # Resilient benchmarking script comparing various optimization approaches
├── train_base_agent_rl_bo.py   # Reinforcement learning agents trained on typical solar cell compositions
├── rl_bo_pipeline/
│   ├── tri_loop.py             # Tri-level optimization loop (BO-RL integration)
│   ├── physics_informed_bo.py  # Bayesian optimization aided by physics-informed surrogates
│   ├── evaluate_fixed_comp.py  # Test of RL agents on fixed compositions
├── data
│   ├── cluster-perovskite-data # MIT comprehensive experimental dataset
│   ├── perovskitedatabase_data # Literature-based dataset (NREL)
│   └── heldout_testdata        # Surrogate model independent validation dataset
├── models/
│   ├── surrogate.pt            # Pre-trained surrogate checkpoint
│   ├── hybrid_efficiency_surrogate.pt    # Efficiency surrogate model
│   └── hybrid_lifetime_surrogate.pt      # Lifetime surrogate model
├── results/
│   └── benchmark_results.csv   # Detailed benchmarking outcomes
└── train_hybrid_surrogate.py   # Complete training pipeline for surrogate models
```

---

## Key Contributions & Workflow

### 1. Physics-Informed Surrogate Modeling

* **Objective:** Cost-effective prediction of solar cell efficiency and lifetime without extensive experiments.
* **Features:** Composition, layer structures, process parameters (spin speed, annealing temperature, encapsulation).
* **Outputs:** Estimated cell efficiency and stability (log-lifetime).
* **Methodology:** High-performance neural network surrogate, which was trained on synthetic physics simulations (SCAPS, Quantum ESPRESSO proxies) and tested against experimental MIT datasets.

### 2. Bayesian Optimization (BO)

* **Objective:** Methodically scan new compositions to discover promising candidates.
* **Method:**

  * Gaussian Process regression with focused acquisition functions (e.g., Expected Improvement).
  * Use of physics-informed priors to make more realistic prediction abilities.

### 3. Reinforcement Learning (RL)

* **Goal:** Accurately optimize fabrication parameters (e.g., spin coating rate, annealing temperature).
* **Procedure:** Soft Actor-Critic (SAC) algorithm learned on compositions proposed by BO to maximize efficiency × lifetime.

### 4. Integrated Tri-Loop Optimization

* Iterative feedback loop: Bayesian optimization proposes compositions → Reinforcement learning optimizes fabrication processes → Surrogate evaluation guides next optimization cycles.
* Progressive autonomous refinement and continuous improvement toward stable, high-efficiency solar cells.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/samy0444/optimization-framework.git
cd optimization-framework
pip install -r requirements.txt
```

---

## Usage

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

### Dashboard Capabilities

* **Manual Simulation:** Quickly evaluate individual solar cell recipes.
* **Automated Optimization:** Run autonomous optimization loops based on BO and RL.
* **Comprehensive Benchmarking:** Run statistical and visual examination of various optimization approaches.

---

## Robust Benchmarking

The benchmarking rigorously compares optimization strategies to ensure statistical significance and robustness:

* Random Search
* Bayesian Optimization (BO-only)
* Reinforcement Learning (RL-only)
* Tri-Loop Optimization (BO+RL)

### Benchmark Methodology

* Make several independent repetitions for robust statistics.
* Report mean performance, standard deviation, and statistical significance test (t-tests).
* Measures: convergence rate, realized efficiency, and lifetime stability.

### Benchmark Visualization

Streamlit-based visualization capabilities:

* Error-band convergence plots.
* Statistical comparison and significance tests.
* Analysis of cycles-to-threshold performance.

---


## Pipeline Workflow Diagram

```
+-------------------+      +---------------------------+
|  Composition BO   | ---> |  Reinforcement Learning   |
| (Gaussian Process)| <--- |     (SAC Fine-tuning)     |
+-------------------+      +---------------------------+
         ^                              |
         |                              v
+-------------------+      +---------------------------+
|  Physics-Informed | ---> |    Fabrication Process    |
| Surrogate Network |      | (Spin Speed, Annealing..) |
+-------------------+      +---------------------------+
```

## Surrogate Model Validation

High-quality validation provides solid generalization:

| Surrogate      | RMSE  | R²    |
| -------------- | ----- | ----- |
| **Lifetime**   | 0.108 | 0.985 |
| **Efficiency** | 3.728 | 0.712 |

* **Lifetime surrogate** results in nearly perfect overall generalization with R² \~ 0.99.
* **Efficiency surrogate** gives stable predictions with RMSE \~3.7% and R² \~ 0.71, considerably high considering inherent experimental volatility.

---

## Benchmark Results
Benchmarking results (shown below) demonstrate that our integrated tri-loop framework (BO + RL) consistently outperforms standalone BO, RL, and random search. Over repeated trials, Tri-Loop achieved superior convergence in optimizing efficiency × lifetime product, with statistically robust performance and minimal variance, highlighting the power of coordinated exploration-exploitation in materials discovery.

Detailed visual and tabulated results showcase optimization efficacy:
![alt text](<visualization (2).png>)

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.


## Contact

- **Email**: samriddhi100mit@gmail.com
