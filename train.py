"""
Walker2D Multi-step DRL Experiment
- Logic 100% theo paper (Meng et al., 2026)
- Tối ưu tốc độ: PPO dùng SubprocVecEnv, off-policy giữ nguyên paper
- GPU-friendly: device=cuda cho neural network updates
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

# ─── Config giống paper ───────────────────────────────────────────
ENV_ID          = "Walker2d-v4"
ENV_TYPES       = ["MDP", "RN", "FLK", "RSM", "RV"]
ALGORITHMS      = {"PPO": None, "TD3": None, "SAC": None, "MTD3": None, "MSAC": None}

TOTAL_TIMESTEPS = 2_500_000
N_SEEDS         = 3
N_EVAL_EPISODES = 5        # paper dùng 5 episodes để eval
EVAL_FREQ       = 10_000   # paper eval mỗi chunk
POLICY_KWARGS   = dict(net_arch=[256, 256])  # paper: 256x256
N_STEP          = 5        # paper: n=5 cho MTD3/MSAC

# ─── Tối ưu tốc độ (không thay đổi logic paper) ──────────────────
# PPO: SubprocVecEnv → collect data song song, KHÔNG thay đổi algorithm
NUM_ENVS_PPO = 4

# Off-policy: giữ ĐÚNG paper hyperparams
# gradient_steps=1, batch_size paper chuẩn
# GPU sẽ tăng tốc neural network updates tự động


def make_vec_env(algo_name, pomdp_type, seed, num_envs):
    if num_envs > 1:
        return SubprocVecEnv([
            make_env_fn(ENV_ID, pomdp_type, seed + i)
            for i in range(num_envs)
        ])
    return DummyVecEnv([make_env_fn(ENV_ID, pomdp_type, seed)])


def create_algorithm(algo_name: str, env, seed: int, device: str = "cpu"):
    """
    Hyperparameters ĐÚNG theo paper (OpenAI SpinningUp standard).
    GPU device → neural network updates nhanh hơn tự động.
    """
    common = dict(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        device=device,      # cuda trên Kaggle → nhanh hơn
        verbose=0,
        policy_kwargs=POLICY_KWARGS,
    )

    if algo_name == "PPO":
        # Paper: GAE lambda=0.97, clip=0.2, n_steps dùng rollout buffer
        # SubprocVecEnv 4 envs → tổng 4*2048=8192 steps/update
        # Không thay đổi algorithm, chỉ thu thập data nhanh hơn
        return PPO(
            **common,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.97,   # paper
            clip_range=0.2,    # paper ε=0.2
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )

    elif algo_name == "TD3":
        # Paper hyperparams: SpinningUp TD3
        return TD3(
            **common,
            learning_rate=1e-3,
            buffer_size=1_000_000,
            learning_starts=10_000,  # paper
            batch_size=100,          # paper
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,        # paper: 1-step bootstrapping
            n_steps=1,
            policy_delay=2,          # paper
            target_policy_noise=0.2, # paper
            target_noise_clip=0.5,   # paper
        )

    elif algo_name == "SAC":
        # Paper hyperparams: SpinningUp SAC
        return SAC(
            **common,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,  # paper
            batch_size=256,          # paper
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,        # paper: 1-step bootstrapping
            n_steps=1,
            ent_coef="auto",         # paper: auto entropy
            target_update_interval=1,
            target_entropy="auto",
        )

    elif algo_name == "MTD3":
        # MTD3: TD3 + n_steps=5 (paper Eq.15)
        return MTD3(
            **common,
            learning_rate=1e-3,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=100,          # same as TD3
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            n_step=N_STEP,           # n=5, paper
        )

    elif algo_name == "MSAC":
        # MSAC: SAC + n_steps=5 (paper Eq.15)
        return MSAC(
            **common,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,          # same as SAC
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            target_update_interval=1,
            target_entropy="auto",
            n_step=N_STEP,           # n=5, paper
        )

    raise ValueError(f"Unknown: {algo_name}")


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

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 500:
        print(f"[SKIP] {algo_name} {env_type} seed={seed}")
        return None

    if os.path.exists(csv_path):
        os.remove(csv_path)

    is_ppo   = (algo_name == "PPO")
    num_envs = NUM_ENVS_PPO if is_ppo else 1

    print(f"\n{'='*60}")
    print(f"Train: {algo_name} | {env_type} | seed={seed} | "
          f"device={device} | envs={num_envs}")
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
        n_eval_episodes=N_EVAL_EPISODES,  # 5 episodes như paper
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
    print(f"# Walker2D Experiment — Logic theo paper (Meng et al. 2026)")
    print(f"# Algorithms    : {algorithms}")
    print(f"# Envs          : {env_types}")
    print(f"# Seeds         : {seeds}")
    print(f"# Total runs    : {total}")
    print(f"# Steps/run     : {total_timesteps:,}")
    print(f"# Device        : {device}")
    print(f"# PPO           : {NUM_ENVS_PPO} envs SubprocVecEnv (speed only)")
    print(f"# Off-policy    : 1 env, gradient_steps=1 (paper exact)")
    print(f"# Eval episodes : {N_EVAL_EPISODES} (paper)")
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