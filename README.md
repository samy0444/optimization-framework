# Self-Driving Solar Lab: Autonomous Optimization Framework for Perovskite Solar Cells

---

## Overview

This repository presents a sophisticated, autonomous optimization framework designed for perovskite solar cell fabrication. It integrates cutting-edge techniques such as physics-informed surrogate modeling, Bayesian optimization (BO), reinforcement learning (RL), and generative AI to simulate a self-driving laboratory. The framework autonomously identifies optimal, stable, and high-efficiency solar cell compositions and fabrication processes, significantly accelerating discovery and development.

---

## Repository Structure

```
├── app.py                      # Streamlit UI for interactive optimization and benchmarking
├── layers.py                   # Solar cell layer properties and configurations
├── interfaces.py               # Interface recombination parameters
├── simulation.py               # Physics-informed surrogate simulation model
├── benchmark_rl_bo.py          # Robust benchmarking script comparing multiple optimization methods
├── train_base_agent_rl_bo.py   # Reinforcement learning agents trained on representative solar cell compositions
├── rl_bo_pipeline/
│   ├── tri_loop.py             # Tri-level optimization loop (BO-RL integration)
│   ├── physics_informed_bo.py  # Bayesian optimization enhanced by physics-informed surrogates
│   ├── evaluate_fixed_comp.py  # Evaluation of RL agents on predefined compositions
├── data
│   ├── cluster-perovskite-data # Comprehensive experimental dataset (MIT)
│   ├── perovskitedatabase_data # Literature-sourced dataset (NREL)
│   └── heldout_testdata        # Independent validation dataset for surrogate model
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

* **Objective:** Efficiently predict solar cell efficiency and lifetime without exhaustive experiments.
* **Features:** Composition, layer structures, process parameters (spin speed, annealing temperature, encapsulation).
* **Outputs:** Predicted cell efficiency and stability (log-lifetime).
* **Methodology:** Advanced neural network surrogate trained on synthetic physics simulations (SCAPS, Quantum ESPRESSO proxies) and validated with real-world MIT experimental datasets.

### 2. Bayesian Optimization (BO)

* **Goal:** Systematically explore novel compositions to identify promising candidates.
* **Approach:**

  * Gaussian Process regression using targeted acquisition functions (e.g., Expected Improvement).
  * Integration of physics-informed priors to enhance realistic prediction capabilities.

### 3. Reinforcement Learning (RL)

* **Purpose:** Precisely optimize fabrication parameters (e.g., spin coating speed, annealing temperature).
* **Method:** Soft Actor-Critic (SAC) algorithm trained on compositions recommended by BO to maximize efficiency × lifetime.

### 4. Integrated Tri-Loop Optimization

* Iterative feedback loop: Bayesian optimization suggests compositions → Reinforcement learning refines fabrication processes → Surrogate evaluation informs subsequent optimization cycles.
* Continuous autonomous refinement and progressive performance enhancement toward stable, high-efficiency solar cells.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/samy0444/self-driving-solar-lab.git
cd self-driving-solar-lab
pip install -r requirements.txt
```

---

## Usage

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

### Dashboard Capabilities

* **Manual Simulation:** Rapidly assess individual solar cell recipes.
* **Automated Optimization:** Perform autonomous optimization loops combining BO and RL.
* **Comprehensive Benchmarking:** Execute statistical and visual analysis of different optimization strategies.

---

## Robust Benchmarking

The benchmarking rigorously compares optimization strategies, ensuring statistical significance and robustness:

* Random Search
* Bayesian Optimization (BO-only)
* Reinforcement Learning (RL-only)
* Tri-Loop Optimization (combined BO and RL)

### Benchmark Methodology

* Conduct multiple independent repetitions for reliable statistics.
* Present mean performance, standard deviation, and statistical significance analysis (t-tests).
* Metrics: convergence rate, achieved efficiency, and lifetime stability.

### Benchmark Visualization

Streamlit-based visualization features:

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

---

## Surrogate Model Validation

Rigorous validation ensures strong generalization:

| Surrogate      | RMSE  | R²    |
| -------------- | ----- | ----- |
| **Lifetime**   | 0.108 | 0.985 |
| **Efficiency** | 3.728 | 0.712 |

* **Lifetime surrogate** achieves near-perfect generalization with R² \~ 0.99.
* **Efficiency surrogate** provides robust predictions with RMSE \~3.7% and R² \~0.71, notably strong given inherent experimental variability.

---

## Benchmark Results

Detailed visual and tabulated results showcase optimization efficacy:
![alt text](<visualization (2).png>)

