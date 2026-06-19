"""
Training utilities for Walker2D experiments.
Fixed: CSV chỉ tạo khi có data thật, không tạo file header rỗng.
"""

import os
import csv
import time
import numpy as np
from typing import List, Tuple
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback


class CSVLoggingCallback(BaseCallback):
    """
    Callback logging training progress to CSV.
    FILE CHỈ ĐƯỢC TẠO KHI CÓ DATA THẬT — không tạo file header rỗng.
    """

    def __init__(
        self,
        eval_env: gym.Env,
        csv_path: str,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        algorithm_name: str = "Unknown",
        env_type: str = "MDP",
        seed: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.csv_path = csv_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.algorithm_name = algorithm_name
        self.env_type = env_type
        self.seed = seed
        self.results = []
        self._last_eval_step = 0
        self._header_written = False  # chỉ write header khi có data

    def _write_row(self, row):
        """Ghi row vào CSV, tạo file + header nếu chưa có."""
        os.makedirs(os.path.dirname(self.csv_path) if os.path.dirname(self.csv_path) else '.', exist_ok=True)
        
        if not self._header_written:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestep', 'mean_return', 'std_return',
                    'max_return', 'min_return',
                    'algorithm', 'env_type', 'seed'
                ])
                writer.writerow(row)
            self._header_written = True
        else:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = self.num_timesteps

            # Evaluate policy
            episode_rewards = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                ep_reward = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                episode_rewards.append(ep_reward)

            mean_r = float(np.mean(episode_rewards))
            std_r  = float(np.std(episode_rewards))
            max_r  = float(np.max(episode_rewards))
            min_r  = float(np.min(episode_rewards))

            row = [
                self.num_timesteps,
                mean_r, std_r, max_r, min_r,
                self.algorithm_name,
                self.env_type,
                self.seed
            ]
            self.results.append(row)
            self._write_row(row)

            if self.verbose > 0:
                print(
                    f"[{self.algorithm_name}|{self.env_type}|seed={self.seed}] "
                    f"Step {self.num_timesteps}: mean_return={mean_r:.1f}"
                )

        return True


def train_agent(
    model,
    env,
    eval_env,
    total_timesteps: int,
    csv_path: str,
    algorithm_name: str,
    env_type: str,
    seed: int,
    eval_freq: int = 5000,
    n_eval_episodes: int = 5,
    verbose: int = 1,
) -> Tuple[list, float]:
    """Train agent với CSV logging. File CSV chỉ tạo khi có data."""

    callback = CSVLoggingCallback(
        eval_env=eval_env,
        csv_path=csv_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        algorithm_name=algorithm_name,
        env_type=env_type,
        seed=seed,
        verbose=verbose,
    )

    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=(verbose > 0),
    )
    training_time = time.time() - start_time

    return callback.results, training_time


def get_max_return(results: list) -> float:
    if not results:
        return 0.0
    return max(r[1] for r in results)


def merge_csv_results(csv_files: List[str], output_path: str) -> None:
    """Merge CSV files — skip file rỗng hoặc lỗi."""
    import pandas as pd

    dfs = []
    for f in csv_files:
        if not os.path.exists(f):
            continue
        try:
            if os.path.getsize(f) < 100:  # skip file quá nhỏ
                continue
            df = pd.read_csv(f, encoding='utf-8')
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"  Skip {f}: {e}")

    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Merged {len(dfs)} files -> {output_path}")
    else:
        print("Không có file CSV hợp lệ nào.")
