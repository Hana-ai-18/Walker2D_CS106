# Walker2D Multi-step DRL Experiment

Replication of key Walker2D experiments from:

> **"Multi-step first: A lightweight deep reinforcement learning strategy for robust continuous control with partial observability"**  
> Lingheng Meng, Rob Gorbet, Michael Burke, Dana Kulić  
> *Neural Networks 199 (2026) 108521*

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| **PPO** | Proximal Policy Optimization (multi-step via GAE) |
| **TD3** | Twin Delayed DDPG (1-step Q-targets) |
| **SAC** | Soft Actor-Critic (1-step Q-targets) |
| **MTD3(5)** | Multi-step TD3, n=5 step bootstrapping |
| **MSAC(5)** | Multi-step SAC, n=5 step bootstrapping |

## Environments

| Type | Description |
|------|-------------|
| **MDP** | Original fully-observable Walker2d-v4 |
| **POMDP-RN** | Random Gaussian noise N(0, 0.01) on all observations |
| **POMDP-FLK** | Flickering: zero entire obs with p=0.2 |
| **POMDP-RSM** | Random sensor missing: zero individual entries with p=0.1 |
| **POMDP-RV** | Remove velocity entries from observation |

## Project Structure

```
walker2d_experiment/
├── train.py              # Main training script
├── simulate.py           # Visualize/render trained agents
├── plot_results.py       # Generate plots from CSV results
├── requirements.txt
├── algorithms/
│   ├── mtd3.py           # Multi-step TD3 (MTD3)
│   ├── msac.py           # Multi-step SAC (MSAC)
│   └── multistep_buffer.py  # n-step replay buffer
├── envs/
│   └── pomdp_wrappers.py # POMDP environment wrappers
├── utils/
│   └── training_utils.py # Training loop + CSV logging
├── results/              # CSV data files (auto-created)
├── models/               # Saved model weights (auto-created)
└── plots/                # Generated plots (auto-created)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train agents

```bash
# Full experiment (all algos, all envs, 3 seeds, 2.5M steps each)
python train.py

# Quick test (50k steps, PPO+SAC, MDP+RN)
python train.py --quick_test

# Single run: PPO on POMDP-RN, seed 0
python train.py --algo PPO --env_type RN --seed 0

# Multi-step on noisy env, 3 seeds
python train.py --algo MSAC --env_type RN --n_seeds 3

# Use GPU
python train.py --device cuda
```

### 2. Simulate trained agents

```bash
# Render PPO on MDP (opens MuJoCo viewer)
python simulate.py --algo PPO --env_type MDP

# Compare all algorithms on POMDP-RN (no render, prints scores)
python simulate.py --compare --env_type RN

# Record video
python simulate.py --algo MSAC --env_type RN --record --output msac_rn.mp4
```

### 3. Plot results

```bash
# Plot all results
python plot_results.py

# Plot specific env type
python plot_results.py --env_type RN

# Custom output dir
python plot_results.py --output_dir my_plots/
```

## Results Format

Each training run produces a CSV file in `results/`:

```
timestep, mean_return, std_return, max_return, min_return, algorithm, env_type, seed
10000, 145.3, 22.1, 178.2, 112.4, PPO, MDP, 0
20000, 287.6, 31.5, 325.1, 251.2, PPO, MDP, 0
...
```

A merged `results/all_results.csv` is created after training completes.

## Key Findings (from paper)

- On **MDP**: TD3 > SAC > PPO (standard result)
- On **POMDPs**: PPO > TD3, PPO > SAC (performance inversion!)
- MTD3(5) improves TD3 on 14/16 POMDP tasks
- MSAC(5) improves SAC on 15/16 POMDP tasks
- The "performance inversion" (PPO winning on POMDPs) is a diagnostic signal of partial observability

## Hardware Notes

- CPU training works fine (recommended for reproducibility)
- Each 2.5M-step run takes ~15-45 minutes depending on algorithm and hardware
- Full experiment (5 algos × 5 envs × 3 seeds = 75 runs) ~2-3 days on CPU
- Use `--total_steps 500000` for faster exploration of results
- Use `--n_seeds 1` for quick single-seed runs

## CPU-based Parallelism

Run multiple seeds/configs in parallel using bash:
```bash
for seed in 0 1 2; do
    python train.py --algo PPO --env_type RN --seed $seed &
done
wait
```

Or use the built-in multiprocessing (adjust based on your CPU count):
```bash
python train.py --algo PPO --env_type RN --n_seeds 3
```
