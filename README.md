# Continual Reinforcement Learning for Adaptive OTEC Control

## Overview

This project explores the use of **continual reinforcement learning** to control an **Ocean Thermal Energy Conversion (OTEC)** system under changing environmental conditions.

OTEC systems generate electricity using the temperature difference between **warm surface seawater and cold deep seawater**. However, **Sea Surface Temperature (SST) varies across seasons and geographical locations**, creating a non-stationary environment for system control.

To address this challenge, this project implements a reinforcement learning controller using **Proximal Policy Optimization (PPO)** and extends it with **Elastic Weight Consolidation (EWC)** to mitigate **catastrophic forgetting** during sequential learning across different SST regimes.

Unlike many simulation-only studies, this work uses **real-world satellite-derived SST datasets stored in NetCDF format**, enabling a more realistic representation of environmental variability affecting OTEC performance.

The goal is to enable an adaptive controller that can **maintain stable power generation even when ocean thermal conditions change over time**.

---

# Dataset

This project uses **real sea surface temperature (SST) data stored in NetCDF (.nc) format**, obtained from oceanographic satellite observations.

The dataset represents **real environmental SST measurements**, which are processed and partitioned into different seasonal regimes used as sequential training tasks for the reinforcement learning agent.

### Available SST Regimes

```
data/
├── summer_location.nc
├── winter_location.nc
├── rainy_location.nc
└── spring_location.nc
```

Each NetCDF file contains:

* Sea Surface Temperature (SST)
* Latitude and Longitude coordinates
* Mean SST statistics
* Seasonal regime metadata

These regimes represent **different ocean thermal operating conditions** under which the OTEC controller must adapt.

---

# Key Features

• Custom **Gymnasium reinforcement learning environment** simulating an open-cycle OTEC system
• Integration of **real satellite-derived SST data (NetCDF)**
• Sequential training across multiple SST regimes
• Implementation of **PPO baseline reinforcement learning agent**
• Integration of **Elastic Weight Consolidation (EWC)** for continual learning
• Evaluation of **catastrophic forgetting in RL control policies**
• Visualization of power retention and performance across environmental regimes

---

# Project Structure

```
continual-rl-otec-control
│
├── environment/
│   └── env_otec.py
│
├── data/
│   ├── summer_location.nc
│   ├── winter_location.nc
│   ├── rainy_location.nc
│   ├── spring_location.nc
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
├── docs/
│   ├── otec_block_diagram.png
│   └── system_architecture.png
│
├── requirements.txt
└── README.md
```

---

# Methodology

The learning framework follows a **continual reinforcement learning setup**, where the agent encounters multiple SST regimes sequentially.

### Training order

```
Summer → Winter → Rainy → Spring
```

Two configurations are compared:

1. **Baseline PPO**
2. **PPO + Elastic Weight Consolidation (EWC)**

Elastic Weight Consolidation estimates parameter importance using the **Fisher Information Matrix** and applies a regularization penalty to prevent critical parameters from drifting during training on new regimes.

This allows the controller to **retain previously learned knowledge while adapting to new environmental conditions**.

---

# Technologies Used

• Python
• PyTorch
• Stable-Baselines3
• Gymnasium
• NumPy
• Pandas
• Xarray
• Matplotlib
• NetCDF4

---

# Installation

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

# Running Experiments

Train the baseline PPO agent:

```
python training/step1_ppo_lifelong.py
```

Train the continual learning agent with EWC:

```
python training/step2_ppo_ewc_lifelong.py
```

Generate performance visualizations:

```
python analysis/visualize_results.py
```

Compute catastrophic forgetting metrics:

```
python analysis/compute_forgetting.py
```

---

# Author

**Vamsi Krishna Gondu**

B.Tech Computer Science and Engineering
Artificial Intelligence & Intelligent Process Automation

KL University, India

---

# License

This project is released under the **MIT License**, allowing open use and modification with proper attribution.
