"""
Walker2D Multi-step DRL Experiment - OPTIMIZED VERSION
- PPO: SubprocVecEnv (4 envs) → collect data song song
- Off-policy: DummyVecEnv (1 env) + gradient_steps=4 → update nhiều hơn/step
- learning_starts nhỏ hơn → bắt đầu học sớm hơn
"""

import os, sys, argparse, time
import numpy as np
import torch
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from envs.pomdp_wrappers import make_env, make_env_fn
from algorithms.mtd3 import MTD3
from algorithms.msac import MSAC
from utils.training_utils import train_agent, get_max_return, merge_csv_results

# ─── Config ───────────────────────────────────────────────────────
ENV_ID       = "Walker2d-v4"
ENV_TYPES    = ["MDP", "RN", "FLK", "RSM", "RV"]
ALGORITHMS   = {"PPO": None, "TD3": None, "SAC": None, "MTD3": None, "MSAC": None}

TOTAL_TIMESTEPS = 2_500_000
N_SEEDS         = 3
N_EVAL_EPISODES = 5
EVAL_FREQ       = 5_000
POLICY_KWARGS   = dict(net_arch=[256, 256])
N_STEP          = 5

# ─── Tối ưu tốc độ ───────────────────────────────────────────────
# PPO: on-policy → nhiều envs = collect data nhanh hơn trực tiếp
NUM_ENVS_PPO = 4          # 4 envs song song cho PPO

# Off-policy: bottleneck là gradient update không phải env step
# → dùng gradient_steps cao thay vì nhiều envs
GRADIENT_STEPS_OFFPOLICY = 4   # update 4 lần/env step
BATCH_SIZE_TD3  = 256          # tăng batch size → tận dụng CPU tốt hơn
BATCH_SIZE_SAC  = 512          # SAC benefit từ large batch


def make_vec_env(algo_name, pomdp_type, seed, num_envs):
    if num_envs > 1:
        return SubprocVecEnv([
            make_env_fn(ENV_ID, pomdp_type, seed + i)
            for i in range(num_envs)
        ])
    else:
        return DummyVecEnv([make_env_fn(ENV_ID, pomdp_type, seed)])


def create_algorithm(algo_name: str, env, seed: int, device: str = "cpu"):
    common = dict(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        device=device,
        verbose=0,
        policy_kwargs=POLICY_KWARGS,
    )

    if algo_name == "PPO":
        # PPO: on-policy, dùng 4 envs → tổng data/update = 4 * n_steps
        return PPO(
            **common,
            learning_rate=3e-4,
            n_steps=2048,       # per env → tổng 4*2048=8192 steps/update
            batch_size=256,     # tăng batch size cho 4 envs
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.97,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )

    elif algo_name == "TD3":
        return TD3(
            **common,
            learning_rate=1e-3,
            buffer_size=1_000_000,
            learning_starts=5_000,              # bắt đầu học sớm hơn
            batch_size=BATCH_SIZE_TD3,          # batch lớn hơn
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=GRADIENT_STEPS_OFFPOLICY,  # 4 updates/step
            n_steps=1,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
        )

    elif algo_name == "SAC":
        return SAC(
            **common,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=5_000,
            batch_size=BATCH_SIZE_SAC,          # batch lớn hơn
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=GRADIENT_STEPS_OFFPOLICY,
            n_steps=1,
            ent_coef="auto",
            target_update_interval=1,
            target_entropy="auto",
        )

    elif algo_name == "MTD3":
        return MTD3(
            **common,
            learning_rate=1e-3,
            buffer_size=1_000_000,
            learning_starts=5_000,
            batch_size=BATCH_SIZE_TD3,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=GRADIENT_STEPS_OFFPOLICY,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            n_step=N_STEP,
        )

    elif algo_name == "MSAC":
        return MSAC(
            **common,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=5_000,
            batch_size=BATCH_SIZE_SAC,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=GRADIENT_STEPS_OFFPOLICY,
            ent_coef="auto",
            target_update_interval=1,
            target_entropy="auto",
            n_step=N_STEP,
        )

    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def run_experiment(
    algo_name, env_type, seed,
    results_dir, models_dir,
    total_timesteps=TOTAL_TIMESTEPS,
    device="cpu", verbose=1,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    pomdp_type = None if env_type == "MDP" else env_type
    csv_path   = os.path.join(results_dir, f"{algo_name}_{env_type}_seed{seed}.csv")
    model_path = os.path.join(models_dir,  f"{algo_name}_{env_type}_seed{seed}")

    # Skip nếu đã có đủ data
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 500:
        print(f"[SKIP] {algo_name} {env_type} seed={seed}")
        return None

    # Xóa file rỗng nếu có
    if os.path.exists(csv_path):
        os.remove(csv_path)

    is_ppo   = (algo_name == "PPO")
    num_envs = NUM_ENVS_PPO if is_ppo else 1

    print(f"\n{'='*60}")
    print(f"Train: {algo_name} | {env_type} | seed={seed} | "
          f"envs={num_envs} | grad_steps={'N/A' if is_ppo else GRADIENT_STEPS_OFFPOLICY}")
    print(f"{'='*60}")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir,  exist_ok=True)

    train_env = make_vec_env(algo_name, pomdp_type, seed, num_envs)
    eval_env  = make_env(ENV_ID, pomdp_type=pomdp_type, seed=seed + 1000)
    model     = create_algorithm(algo_name, train_env, seed=seed, device=device)

    t0 = time.time()
    results, _ = train_agent(
        model=model,
        env=train_env,
        eval_env=eval_env,
        total_timesteps=total_timesteps,
        csv_path=csv_path,
        algorithm_name=algo_name,
        env_type=env_type,
        seed=seed,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    model.save(model_path)
    train_env.close()
    eval_env.close()

    max_ret = get_max_return(results)
    print(f"✓ Done! max_return={max_ret:.1f} | {elapsed/60:.1f} phút")
    return max_ret


def run_full_experiment(
    algorithms, env_types, seeds,
    results_dir="results", models_dir="models",
    total_timesteps=TOTAL_TIMESTEPS, device="cpu", verbose=1,
):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir,  exist_ok=True)

    total = len(algorithms) * len(env_types) * len(seeds)
    cur   = 0
    summary = {}

    print(f"\n{'#'*65}")
    print(f"# Walker2D Experiment - OPTIMIZED")
    print(f"# Algorithms       : {algorithms}")
    print(f"# Envs             : {env_types}")
    print(f"# Seeds            : {seeds}")
    print(f"# Total runs       : {total}")
    print(f"# Steps/run        : {total_timesteps:,}")
    print(f"# PPO              : {NUM_ENVS_PPO} envs (SubprocVecEnv)")
    print(f"# Off-policy       : 1 env + gradient_steps={GRADIENT_STEPS_OFFPOLICY}")
    print(f"{'#'*65}\n")

    for env_type in env_types:
        for algo in algorithms:
            for seed in seeds:
                cur += 1
                print(f"\n[{cur}/{total}]")
                r = run_experiment(algo, env_type, seed,
                                   results_dir, models_dir,
                                   total_timesteps, device, verbose)
                key = f"{algo}_{env_type}"
                if key not in summary: summary[key] = []
                if r: summary[key].append(r)

    import glob
    csvs = [f for f in glob.glob(os.path.join(results_dir, "*.csv"))
            if "all_results" not in f]
    merge_csv_results(csvs, os.path.join(results_dir, "all_results.csv"))
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo",        type=str, default=None)
    p.add_argument("--env_type",    type=str, default=None)
    p.add_argument("--seed",        type=int, default=None)
    p.add_argument("--n_seeds",     type=int, default=3)
    p.add_argument("--total_steps", type=int, default=TOTAL_TIMESTEPS)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--models_dir",  type=str, default="models")
    p.add_argument("--device",      type=str, default="cpu")
    p.add_argument("--verbose",     type=int, default=1)
    p.add_argument("--quick_test",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.quick_test:
        run_full_experiment(["PPO","SAC"], ["MDP","RN"], [0],
                            total_timesteps=30_000, device=args.device)
        sys.exit(0)

    algos     = list(ALGORITHMS.keys()) if not args.algo     else [args.algo]
    env_types = ENV_TYPES               if not args.env_type else [args.env_type]
    seeds     = list(range(args.n_seeds)) if args.seed is None else [args.seed]

    run_full_experiment(algos, env_types, seeds,
                        args.results_dir, args.models_dir,
                        args.total_steps, args.device, args.verbose)
