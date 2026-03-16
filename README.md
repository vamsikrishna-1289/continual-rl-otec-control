# Continual Reinforcement Learning for Adaptive OTEC Control

## Overview

This project explores the use of **continual reinforcement learning** to control an **Ocean Thermal Energy Conversion (OTEC)** system under changing environmental conditions. OTEC systems generate electricity using the temperature difference between warm surface seawater and cold deep seawater. However, **sea surface temperature (SST) varies across seasons and locations**, creating a non-stationary environment for system control.

To address this challenge, this project implements a reinforcement learning controller using **Proximal Policy Optimization (PPO)** and extends it with **Elastic Weight Consolidation (EWC)** to reduce catastrophic forgetting during sequential training across multiple SST regimes.

The goal is to enable an adaptive controller that can **maintain stable power generation even when environmental conditions change over time**.

---

## Key Features

* Custom **Gymnasium reinforcement learning environment** simulating an open-cycle OTEC system
* Real **Sea Surface Temperature (SST) datasets stored in NetCDF format**
* Sequential training across multiple SST regimes (Summer, Winter, Rainy, Spring)
* Implementation of **PPO baseline agent**
* Integration of **Elastic Weight Consolidation (EWC)** for continual learning
* Evaluation of **catastrophic forgetting in reinforcement learning controllers**
* Visualization of power retention and policy performance across regimes

---

## Project Structure

```
continual-rl-otec-control
│
├── environment/
│   └── env_otec.py
│
├── data/
│   └── prepare_sst_regimes.py
│
├── training/
│   ├── step1_ppo_lifelong.py
│   └── step2_ppo_ewc_lifelong.py
│
├── analysis/
│   ├── compute_forgetting.py
│   ├── visualize_results.py
│   └── lambda_ablation.py
│
├── results/
│   └── experiment outputs
│
├── figures/
│   └── generated plots
│
├── requirements.txt
└── README.md
```

---

## Methodology

The learning framework follows a **sequential training setup** where the agent encounters multiple SST regimes as separate tasks.

Training order:

```
Summer → Winter → Rainy → Spring
```

Two configurations are compared:

1. **Baseline PPO**
2. **PPO + Elastic Weight Consolidation (EWC)**

EWC estimates parameter importance using the **Fisher Information Matrix** and applies a penalty to prevent important parameters from drifting during new task training.

This allows the policy to **retain previously learned knowledge while adapting to new environmental conditions**.

---

## Technologies Used

* Python
* PyTorch
* Stable-Baselines3
* Gymnasium
* NumPy
* Pandas
* Xarray
* Matplotlib

---

## Installation

Clone the repository:

```
git clone https://github.com/vamsikrishna-1289/continual-rl-otec-control.git
cd continual-rl-otec-control
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running Experiments

Train the baseline PPO agent:

```
python training/step1_ppo_lifelong.py
```

Train the continual learning agent with EWC:

```
python training/step2_ppo_ewc_lifelong.py
```

Generate result visualizations:

```
python analysis/visualize_results.py
```

Compute catastrophic forgetting metrics:

```
python analysis/compute_forgetting.py
```

## Author

**Vamsi Krishna Gondu**

B.Tech Computer Science and Engineering (AI & Intelligent Process Automation)

KL University, India

---

## License

This project is released under the **MIT License**, allowing open use and modification with proper attribution.
