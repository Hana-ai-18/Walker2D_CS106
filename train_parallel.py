"""
Parallel training - Windows safe.

Cả PPO lẫn off-policy đều dùng SubprocVecEnv (NUM_ENVS envs mỗi run).
Mỗi run chiếm ~NUM_ENVS cores → n_workers tính theo đó.

Với 22 cores, NUM_ENVS_OFF=4:
  n_workers = (22 - 2) // 4 = 5 workers song song ✅
"""

import os, sys, argparse, time, subprocess, glob
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import ENV_TYPES, ALGORITHMS, TOTAL_TIMESTEPS, NUM_ENVS_PPO, NUM_ENVS_OFF_POLICY, TRAIN_FREQ_OFF


def run_single(task):
    algo, env_type, seed, results_dir, models_dir, total_steps, num_envs_off = task
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")

    cmd = [
        sys.executable, script,
        "--algo", algo, "--env_type", env_type,
        "--seed", str(seed), "--n_seeds", "1",
        "--total_steps", str(total_steps),
        "--results_dir", results_dir,
        "--models_dir", models_dir,
        "--device", "cuda",
        "--verbose", "0",
        "--num_envs_off", str(num_envs_off),
    ]

    print(f"  START → {algo:5} | {env_type:4} | seed={seed}")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=28800)
        elapsed = time.time() - t0
        if r.returncode != 0:
            print(f"  ERROR → {algo:5} | {env_type:4} | seed={seed}\n{r.stderr[-300:]}")
            return (algo, env_type, seed, False)
        print(f"  DONE  → {algo:5} | {env_type:4} | seed={seed} | {elapsed/60:.0f}m")
        return (algo, env_type, seed, True)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT → {algo:5} | {env_type:4} | seed={seed}")
        return (algo, env_type, seed, False)
    except Exception as e:
        print(f"  FAIL → {algo:5} | {env_type:4} | seed={seed} | {e}")
        return (algo, env_type, seed, False)


def get_pending(algorithms, env_types, seeds, results_dir):
    pending, skipped = [], 0
    for algo in algorithms:
        for env_type in env_types:
            for seed in seeds:
                f = os.path.join(results_dir, f"{algo}_{env_type}_seed{seed}.csv")
                if os.path.exists(f) and os.path.getsize(f) > 500:
                    skipped += 1
                else:
                    if os.path.exists(f):
                        os.remove(f)
                    pending.append((algo, env_type, seed))
    return pending, skipped


def run_parallel(algorithms, env_types, seeds,
                 results_dir="results", models_dir="models",
                 total_steps=500_000, n_workers=None,
                 num_envs_off=NUM_ENVS_OFF_POLICY):

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir,  exist_ok=True)

    pending, skipped = get_pending(algorithms, env_types, seeds, results_dir)
    tasks = [(a, e, s, results_dir, models_dir, total_steps, num_envs_off)
             for a, e, s in pending]

    total_cores = os.cpu_count()
    if n_workers is None:
        # Bottleneck = off-policy runs (NUM_ENVS_OFF_POLICY cores/run)
        n_workers = max(1, (total_cores - 2) // num_envs_off)

    import math
    n_ppo    = sum(1 for a, _, _ in pending if a == "PPO")
    n_others = len(pending) - n_ppo
    est_h    = (math.ceil(n_ppo    / n_workers) * 15 +
                math.ceil(max(1, n_others) / n_workers) * 10) / 60 * (total_steps / 500_000)

    eff = TRAIN_FREQ_OFF * num_envs_off
    print(f"\n{'#'*65}")
    print(f"# Walker2D Parallel — {total_cores} cores")
    print(f"# n_workers          : {n_workers}")
    print(f"# PPO/run            : {NUM_ENVS_PPO} envs SubprocVecEnv")
    print(f"# Off-policy/run     : {num_envs_off} envs SubprocVecEnv")
    print(f"#   train_freq={TRAIN_FREQ_OFF} × {num_envs_off} envs = {eff} trans/update")
    print(f"#   gradient_steps=-1 → auto={eff} → ratio 1:1 paper ✓")
    print(f"# Total tasks        : {len(algorithms)*len(env_types)*len(seeds)}")
    print(f"# Already done       : {skipped} (skip)")
    print(f"# Pending            : {len(tasks)}")
    print(f"# Steps/run          : {total_steps:,}")
    print(f"# Ước tính           : ~{est_h:.1f} giờ")
    print(f"{'#'*65}\n")

    if not tasks:
        print("Tất cả đã xong! Chạy: python plot_results.py")
        return

    start = time.time()
    success, failed = [], []

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(run_single, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                a, e, s, ok = future.result()
                (success if ok else failed).append((a, e, s))
            except Exception as ex_e:
                t = futures[future]
                failed.append((t[0], t[1], t[2]))
            print(f"  [{done}/{len(tasks)} xong]")

    elapsed = time.time() - start
    print(f"\n{'='*65}")
    print(f"HOÀN THÀNH! {elapsed/3600:.1f} giờ | "
          f"Thành công: {len(success)}/{len(tasks)}")
    if failed:
        for a, e, s in failed:
            print(f"  FAIL: {a} {e} seed={s}")
    print(f"{'='*65}")

    import pandas as pd
    csvs = [f for f in glob.glob(os.path.join(results_dir, "*.csv"))
            if "all_results" not in os.path.basename(f)
            and os.path.getsize(f) > 100]
    if csvs:
        df  = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
        out = os.path.join(results_dir, "all_results.csv")
        df.to_csv(out, index=False)
        print(f"\nMerged {len(csvs)} files → {out}")
        print("Chạy: python plot_results.py")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total_steps",  type=int, default=500_000)
    p.add_argument("--n_seeds",      type=int, default=3)
    p.add_argument("--n_workers",    type=int, default=None)
    p.add_argument("--results_dir",  type=str, default="results")
    p.add_argument("--models_dir",   type=str, default="models")
    p.add_argument("--algo",         type=str, default=None)
    p.add_argument("--env_type",     type=str, default=None)
    p.add_argument("--num_envs_off", type=int, default=NUM_ENVS_OFF_POLICY,
                   help="Số envs song song cho off-policy (default: 4)")
    return p.parse_args()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    args  = parse_args()
    algos = list(ALGORITHMS.keys()) if not args.algo     else [args.algo]
    envs  = ENV_TYPES               if not args.env_type else [args.env_type]
    seeds = list(range(args.n_seeds))

    run_parallel(algos, envs, seeds,
                 args.results_dir, args.models_dir,
                 args.total_steps, args.n_workers,
                 args.num_envs_off)
