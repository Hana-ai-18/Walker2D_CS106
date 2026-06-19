"""
MTD3: Multi-step TD3.

SB3 >= 2.0 natively supports n-step returns via the `n_steps` parameter.
The buffer stores n-step discounted rewards and gamma^n as `discounts`.

Usage:
    model = MTD3("MlpPolicy", env, n_steps=5)
"""
from stable_baselines3 import TD3
from stable_baselines3.common.type_aliases import GymEnv
from typing import Optional, Union, Type, Dict, Any, Tuple
import torch
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.td3.policies import TD3Policy


class MTD3(TD3):
    """Multi-step TD3 using SB3's built-in n-step return support."""
    
    def __init__(
        self,
        policy,
        env,
        learning_rate=1e-3,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=100,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=None,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        stats_window_size=100,
        tensorboard_log=None,
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
        n_step: int = 5,
    ):
        # SB3 uses n_steps for n-step returns
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_step,  # SB3 2.9 native n-step support
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self.n_step = n_step
