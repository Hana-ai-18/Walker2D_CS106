"""
Walker2D Multi-step DRL Experiment
- Logic 100% theo paper (Meng et al., 2026)
- Off-policy CŨNG chạy song song nhiều envs như PPO:
    SubprocVecEnv với N_ENVS envs + train_freq=N + gradient_steps=-1
    SB3 tự tính gradient_steps = N * N_ENVS → ratio 1:1 đúng paper ✓
    GPU nhận batch lớn liên tục, không bị idle ✓

Tại sao gradient_steps=-1?
  Mỗi vòng SB3 collect train_freq * n_envs transitions.
  gradient_steps=-1 → SB3 tự set = số transitions vừa collect
  → tự động đúng ratio bất kể train_freq hay n_envs là bao nhiêu ✓
"""

import os, sys, argparse, time
import numpy as np
import torch

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
N_EVAL_EPISODES = 5
EVAL_FREQ       = 10_000
POLICY_KWARGS   = dict(net_arch=[256, 256])
N_STEP          = 5          # paper: n=5 cho MTD3/MSAC

# ─── Song song envs ───────────────────────────────────────────────
# Cả PPO lẫn off-policy đều dùng SubprocVecEnv
NUM_ENVS_PPO        = 4   # PPO: 4 envs song song
NUM_ENVS_OFF_POLICY = 4   # TD3/SAC/MTD3/MSAC: 4 envs song song

# train_freq cho off-policy: collect bao nhiêu steps/env trước mỗi update
# gradient_steps=-1 → SB3 tự tính = train_freq * n_envs → ratio 1:1 tự động ✓
TRAIN_FREQ_OFF = 64   # mỗi vòng collect 64*4=256 transitions → update 256 lần


def make_vec_env(pomdp_type, seed, num_envs):
    if num_envs > 1:
        return SubprocVecEnv([
            make_env_fn(ENV_ID, pomdp_type, seed + i)
            for i in range(num_envs)
        ])
    return DummyVecEnv([make_env_fn(ENV_ID, pomdp_type, seed)])


def create_algorithm(algo_name: str, env, seed: int, device: str = "cpu",
                     num_envs: int = 1):
    """
    Hyperparameters đúng theo paper.
    Off-policy: train_freq=TRAIN_FREQ_OFF, gradient_steps=-1
      → SB3 tự tính gradient_steps = TRAIN_FREQ_OFF * num_envs mỗi vòng
      → ratio 1 gradient/transition đúng paper, GPU không bị idle ✓
    """
    common = dict(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        device=device,
        verbose=0,
        policy_kwargs=POLICY_KWARGS,
    )

    if algo_name == "PPO":
        # n_steps=2048: collect 2048*num_envs steps/update
        return PPO(
            **common,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
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
            learning_starts=10_000,
            batch_size=100,
            tau=0.005,
            gamma=0.99,
            train_freq=TRAIN_FREQ_OFF,
            gradient_steps=-1,          # tự tính = TRAIN_FREQ_OFF * num_envs ✓
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
        )

    elif algo_name == "SAC":
        return SAC(
            **common,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=TRAIN_FREQ_OFF,
            gradient_steps=-1,          # tự tính ✓
            ent_coef="auto",
            target_update_interval=1,
            target_entropy="auto",
        )

    elif algo_name == "MTD3":
        return MTD3(
            **common,
            learning_rate=1e-3,
            buffer_size=1_000_000,
            learning_starts=10_000,
            batch_size=100,
            tau=0.005,
            gamma=0.99,
            train_freq=TRAIN_FREQ_OFF,
            gradient_steps=-1,          # tự tính ✓
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
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=TRAIN_FREQ_OFF,
            gradient_steps=-1,          # tự tính ✓
            ent_coef="auto",
            target_update_interval=1,
            target_entropy="auto",
            n_step=N_STEP,
        )

    raise ValueError(f"Unknown: {algo_name}")


def run_experiment(
    algo_name, env_type, seed,
    results_dir, models_dir,
    total_timesteps=TOTAL_TIMESTEPS,
    device="cpu", verbose=1,
    num_envs_off=NUM_ENVS_OFF_POLICY,
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
    num_envs = NUM_ENVS_PPO if is_ppo else num_envs_off

    print(f"\n{'='*60}")
    print(f"Train: {algo_name} | {env_type} | seed={seed} | device={device}")
    if is_ppo:
        print(f"  PPO: {num_envs} envs SubprocVecEnv, n_steps=2048")
    else:
        effective = TRAIN_FREQ_OFF * num_envs
        print(f"  Off-policy: {num_envs} envs SubprocVecEnv")
        print(f"  train_freq={TRAIN_FREQ_OFF} × {num_envs} envs = {effective} transitions/update")
        print(f"  gradient_steps=-1 → SB3 auto = {effective} ✓")
    print(f"{'='*60}")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir,  exist_ok=True)

    train_env = make_vec_env(pomdp_type, seed, num_envs)
    eval_env  = make_env(ENV_ID, pomdp_type=pomdp_type, seed=seed + 1000)
    model     = create_algorithm(algo_name, train_env, seed=seed,
                                 device=device, num_envs=num_envs)

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
    num_envs_off=NUM_ENVS_OFF_POLICY,
):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir,  exist_ok=True)

    total   = len(algorithms) * len(env_types) * len(seeds)
    cur     = 0
    summary = {}

    eff = TRAIN_FREQ_OFF * num_envs_off
    print("\n" + "="*65)
    print(f"Walker2D Experiment — Logic theo paper (Meng et al. 2026)")
    print(f"  Algorithms    : {algorithms}")
    print(f"  Envs          : {env_types}")
    print(f"  Seeds         : {seeds}")
    print(f"  Total runs    : {total}")
    print(f"  Steps/run     : {total_timesteps:,}")
    print(f"  Device        : {device}")
    print(f"  PPO           : {NUM_ENVS_PPO} envs SubprocVecEnv")
    print(f"  Off-policy    : {num_envs_off} envs SubprocVecEnv")
    print(f"    train_freq  = {TRAIN_FREQ_OFF} × {num_envs_off} envs = {eff} trans/update")
    print(f"    grad_steps  = -1 (auto = {eff}) → ratio 1:1 paper ✓")
    print(f"  Eval episodes : {N_EVAL_EPISODES} (paper)")
    print("="*65 + "\n")

    for env_type in env_types:
        for algo in algorithms:
            for seed in seeds:
                cur += 1
                print(f"\n[{cur}/{total}]")
                r = run_experiment(algo, env_type, seed,
                                   results_dir, models_dir,
                                   total_timesteps, device, verbose,
                                   num_envs_off)
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
    p.add_argument("--algo",           type=str, default=None)
    p.add_argument("--env_type",       type=str, default=None)
    p.add_argument("--seed",           type=int, default=None)
    p.add_argument("--n_seeds",        type=int, default=3)
    p.add_argument("--total_steps",    type=int, default=TOTAL_TIMESTEPS)
    p.add_argument("--results_dir",    type=str, default="results")
    p.add_argument("--models_dir",     type=str, default="models")
    p.add_argument("--device",         type=str, default="cpu")
    p.add_argument("--verbose",        type=int, default=1)
    p.add_argument("--quick_test",     action="store_true")
    p.add_argument("--num_envs_off",   type=int, default=NUM_ENVS_OFF_POLICY,
                   help="Số envs song song cho off-policy (default: 4)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.quick_test:
        run_full_experiment(
            ["PPO", "SAC", "TD3", "MTD3", "MSAC"], ["MDP", "RN"], [0],
            total_timesteps=30_000, device=args.device,
            num_envs_off=args.num_envs_off,
        )
        sys.exit(0)

    algos     = list(ALGORITHMS.keys()) if not args.algo     else [args.algo]
    env_types = ENV_TYPES               if not args.env_type else [args.env_type]
    seeds     = list(range(args.n_seeds)) if args.seed is None else [args.seed]

    run_full_experiment(algos, env_types, seeds,
                        args.results_dir, args.models_dir,
                        args.total_steps, args.device, args.verbose,
                        args.num_envs_off)
