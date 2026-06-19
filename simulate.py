"""
Simulation script: visualize a trained Walker2D agent.

Usage:
    # Render PPO on MDP (default):
    python simulate.py --algo PPO --env_type MDP --seed 0

    # Render MSAC on noisy env:
    python simulate.py --algo MSAC --env_type RN --seed 0

    # Record video:
    python simulate.py --algo PPO --env_type MDP --seed 0 --record --output video.mp4
    
    # Run multiple episodes:
    python simulate.py --algo SAC --env_type FLK --n_episodes 5
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO, TD3, SAC
from envs.pomdp_wrappers import make_env
from algorithms.mtd3 import MTD3
from algorithms.msac import MSAC


ALGO_CLASSES = {
    "PPO": PPO,
    "TD3": TD3,
    "SAC": SAC,
    "MTD3": MTD3,
    "MSAC": MSAC,
}


def run_simulation(
    algo_name: str,
    env_type: str,
    seed: int = 0,
    models_dir: str = "models",
    n_episodes: int = 3,
    record: bool = False,
    output_path: str = "simulation.mp4",
    max_steps: int = 1000,
):
    """Load a trained model and run simulation."""
    
    model_path = os.path.join(
        models_dir, f"{algo_name}_{env_type}_seed{seed}.zip"
    )
    
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print(f"  Run 'python train.py --algo {algo_name} --env_type {env_type} --seed {seed}' first.")
        sys.exit(1)
    
    print(f"\nLoading: {model_path}")
    
    # Load model
    AlgoClass = ALGO_CLASSES[algo_name]
    model = AlgoClass.load(model_path)
    
    # Render mode
    if record:
        render_mode = "rgb_array"
    else:
        render_mode = "human"
    
    pomdp_type = None if env_type == "MDP" else env_type
    env = make_env("Walker2d-v4", pomdp_type=pomdp_type, seed=seed, render_mode=render_mode)
    
    # Video recording setup
    if record:
        try:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            frames = []
            recording = True
        except ImportError:
            print("⚠ moviepy not installed. Install with: pip install moviepy")
            print("  Falling back to display mode.")
            record = False
            recording = False
            env.close()
            env = make_env("Walker2d-v4", pomdp_type=pomdp_type, seed=seed, render_mode="human")
    else:
        recording = False
    
    print(f"\nSimulating {algo_name} on Walker2D-{env_type}")
    print(f"Episodes: {n_episodes} | Max steps/ep: {max_steps}")
    print("-" * 50)
    
    episode_returns = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        step = 0
        
        while step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            step += 1
            
            if recording:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            
            if terminated or truncated:
                break
        
        episode_returns.append(ep_return)
        print(f"  Episode {ep+1}: return={ep_return:.1f}, steps={step}")
    
    print("-" * 50)
    print(f"Mean return: {np.mean(episode_returns):.1f} ± {np.std(episode_returns):.1f}")
    
    # Save video
    if recording and frames:
        print(f"\nSaving video to: {output_path}")
        clip = ImageSequenceClip(frames, fps=30)
        clip.write_videofile(output_path, verbose=False, logger=None)
        print(f"✓ Video saved: {output_path}")
    
    env.close()
    return episode_returns


def compare_agents(
    algorithms: list,
    env_type: str,
    seed: int = 0,
    models_dir: str = "models",
    n_episodes: int = 3,
):
    """Compare multiple trained agents on the same environment."""
    print(f"\n{'='*60}")
    print(f"Comparing agents on Walker2D-{env_type}")
    print(f"{'='*60}")
    
    results = {}
    for algo_name in algorithms:
        model_path = os.path.join(
            models_dir, f"{algo_name}_{env_type}_seed{seed}.zip"
        )
        if not os.path.exists(model_path):
            print(f"[SKIP] {algo_name}: model not found at {model_path}")
            continue
        
        AlgoClass = ALGO_CLASSES[algo_name]
        model = AlgoClass.load(model_path)
        
        pomdp_type = None if env_type == "MDP" else env_type
        env = make_env("Walker2d-v4", pomdp_type=pomdp_type, seed=seed)
        
        episode_returns = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_return = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += reward
                done = terminated or truncated
            episode_returns.append(ep_return)
        
        env.close()
        results[algo_name] = episode_returns
        print(f"  {algo_name:8}: {np.mean(episode_returns):8.1f} ± {np.std(episode_returns):.1f}")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate trained Walker2D agents")
    parser.add_argument("--algo", type=str, default="PPO",
                        choices=["PPO", "TD3", "SAC", "MTD3", "MSAC"])
    parser.add_argument("--env_type", type=str, default="MDP",
                        choices=["MDP", "RN", "FLK", "RSM", "RV"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--record", action="store_true", help="Record video")
    parser.add_argument("--output", type=str, default="simulation.mp4")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--compare", action="store_true",
                        help="Compare all algorithms on given env_type")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.compare:
        compare_agents(
            algorithms=["PPO", "TD3", "SAC", "MTD3", "MSAC"],
            env_type=args.env_type,
            seed=args.seed,
            models_dir=args.models_dir,
            n_episodes=args.n_episodes,
        )
    else:
        run_simulation(
            algo_name=args.algo,
            env_type=args.env_type,
            seed=args.seed,
            models_dir=args.models_dir,
            n_episodes=args.n_episodes,
            record=args.record,
            output_path=args.output,
            max_steps=args.max_steps,
        )
