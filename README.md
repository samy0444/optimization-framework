
# Self-Driving Solar Lab: Autonomous Optimization Framework for Perovskite Solar Cells

---

## Overview

This repository provides a comprehensive framework for autonomous optimization of perovskite solar cell fabrication. Leveraging physics-informed surrogate modeling, Bayesian optimization (BO), reinforcement learning (RL), and generative AI, this project simulates a self-driving laboratory that autonomously discovers stable, high-efficiency solar cell compositions and fabrication processes.

---

## Repository Structure

```
├── app.py                      # Streamlit UI for user interaction and automated optimization
├── layers.py                   # Defines solar cell layer properties and configurations
├── interfaces.py               # Interface recombination parameters
├── simulation.py               # Physics-informed surrogate simulation model
├── benchmark_rl_bo.py          # Robust benchmarking script comparing methods
├── train_base_agent_rl_bo.py   # Training RL agents on representative solar cell compositions
├── rl_bo_pipeline/
│   ├── tri_loop.py             # Orchestrates the tri-level BO-RL optimization loop
│   ├── physics_informed_bo.py  # Bayesian optimization with physics-informed surrogate
│   ├── evaluate_fixed_comp.py  # Evaluates RL agents on fixed compositions
├── data
│   └── cluster-perovskite-data # For more Info  https://github.com/PV-Lab/cluster-perovskite-data.git
│   └── perovskitedatabase_data # For more Info https://github.com/Jesperkemist/perovskitedatabase_data.git
│   └── heldout_testdata        # To Perform Predictions & Compute Metrics using hybrid_surrogate
├── models/
│   ├── surrogate.pt            # Pre-trained surrogate model checkpoint
│   ├── hybrid_efficiency_surrogate.pt    # Trained efficiency surrogate
│   └── hybrid_lifetime_surrogate.pt      # Trained lifetime surrogate
├── results/
│   └── benchmark_results.csv   # Generated benchmarking results
└── train_hybrid_surrogate.py   # Full surrogate training pipeline (efficiency + lifetime)
```

---

## Key Components & Workflow

### 1. Physics-Informed Surrogate Simulation

* **Purpose**: Rapidly predict device efficiency and lifetime without costly experiments.
* **Input**: Composition, layer stack, process parameters (spin speed, annealing temperature, encapsulation).
* **Output**: Predicted efficiency and stability (log-lifetime).
* **Method**: Differentiable surrogate neural network trained on simulation (SCAPS, Quantum ESPRESSO proxies) and real-world MIT dataset.

### 2. Bayesian Optimization (BO)

* **Purpose**: Explore and propose promising new compositions.
* **Approach**:

  * Gaussian Process regression with acquisition functions (e.g., Expected Improvement).
  * Physics-informed priors and penalties for realistic predictions.

### 3. Reinforcement Learning (RL)

* **Purpose**: Fine-tune fabrication processes (spin coating, annealing).
* **Method**: SAC algorithm fine-tuned on BO-suggested compositions to optimize efficiency × lifetime.

### 4. Tri-Level Optimization Loop ("Tri-Loop")

* Iterative cycle: BO → RL fine-tuning → surrogate evaluation → BO update.
* Autonomous continuous improvement toward stable, high-efficiency compositions.

---

## Package Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/samy0444/self-driving-solar-lab.git
cd self-driving-solar-lab
pip install -r requirements.txt
```

---

## Usage

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

### Dashboard Features

* **Manual Fabrication Simulation**: Quickly test individual solar cell recipes.
* **Automated Optimization**: Run automated BO + RL optimization cycles.
* **Robust Benchmarking**: Execute statistical benchmarking with visualized results.

---

## Robust Benchmarking

The benchmarking setup rigorously compares different optimization strategies:

* Random Search
* Bayesian Optimization (BO-only)
* Reinforcement Learning (RL-only)
* Tri-Loop Optimization (BO + RL)

### Benchmarking Process

* Multiple independent repeats for statistical robustness.
* Reports mean performance, standard deviation, and statistical significance (t-tests).
* Metrics include convergence speed, final efficiency, and lifetime stability.

### Benchmark Visualization

Use the provided Streamlit interface to visualize:

* Convergence plots with error bands.
* Statistical comparisons and significance testing.
* Cycles-to-threshold performance analysis.

---

## Diagram: Pipeline Workflow

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

## Surrogate Validation

To ensure that both lifetime and efficiency surrogate models generalize well to unseen data:

1. **Load Held‑Out Test Splits**
2. **Reconstruct Models and Scalers**
3. **Perform Predictions & Compute Metrics**

| Surrogate   | RMSE        | R²             |
|------------ | ----------- | -------------- |
| **Lifetime**   | 0.108       | 0.985          |
| **Efficiency** | 3.728       | 0.712          |

> **Notes:**
> - Lifetime is in log‑lifetime units; R²≈0.99 shows excellent generalization even on real MIT data.  
> - Efficiency RMSE≈3.7% (absolute PCE), R²≈0.71 is strong given the noise in literature‐sourced JV measurements.
