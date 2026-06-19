"""
POMDP Wrappers for Walker2D environments.
Based on: Multi-step first paper (Meng et al., 2026)

Four POMDP variants:
- POMDP-RV: Remove Velocity entries
- POMDP-FLK: Flickering (zero obs with prob p=0.2)
- POMDP-RN: Random Noise (Gaussian, sigma=0.1)
- POMDP-RSM: Random Sensor Missing (zero individual entries, p=0.1)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RemoveVelocityWrapper(gym.ObservationWrapper):
    """POMDP-RV: Remove velocity entries from Walker2D observation.
    
    Walker2D obs (17 dims):
      [0]    z-position of torso
      [1]    x-angle of torso
      [2-8]  joint angles (7 joints)
      [9]    x-velocity of torso
      [10]   z-velocity of torso
      [11]   y-angular velocity of torso
      [12-17] joint velocities (6 joints)
    Velocity dims = indices 9-17 (9 dims), keep positions [0-8] (9 dims)
    """
    def __init__(self, env):
        super().__init__(env)
        # Walker2D: first 9 are positions, last 8 are velocities -> keep first 9
        obs_dim = env.observation_space.shape[0]
        # Detect velocity indices: roughly second half
        # For Walker2d-v4: 17 obs, positions=9, velocities=8
        self.keep_indices = list(range(9))  # keep position obs only
        low = env.observation_space.low[self.keep_indices]
        high = env.observation_space.high[self.keep_indices]
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return obs[self.keep_indices].astype(np.float32)


class FlickeringWrapper(gym.ObservationWrapper):
    """POMDP-FLK: Zero entire observation with probability p_flk=0.2."""
    
    def __init__(self, env, p_flk=0.2):
        super().__init__(env)
        self.p_flk = p_flk
        self._last_obs = np.zeros(env.observation_space.shape, dtype=np.float32)

    def observation(self, obs):
        if np.random.random() < self.p_flk:
            return np.zeros_like(obs, dtype=np.float32)
        self._last_obs = obs.astype(np.float32)
        return self._last_obs


class RandomNoiseWrapper(gym.ObservationWrapper):
    """POMDP-RN: Add Gaussian noise N(0, sigma^2) to observations."""
    
    def __init__(self, env, sigma=0.1):
        super().__init__(env)
        self.sigma = sigma

    def observation(self, obs):
        noise = np.random.normal(0, self.sigma, obs.shape).astype(np.float32)
        return (obs + noise).astype(np.float32)


class RandomSensorMissingWrapper(gym.ObservationWrapper):
    """POMDP-RSM: Zero individual entries with probability p_rsm=0.1."""
    
    def __init__(self, env, p_rsm=0.1):
        super().__init__(env)
        self.p_rsm = p_rsm

    def observation(self, obs):
        obs = obs.astype(np.float32).copy()
        mask = np.random.random(obs.shape) < self.p_rsm
        obs[mask] = 0.0
        return obs


def make_env(env_id="Walker2d-v4", pomdp_type=None, seed=0, render_mode=None):
    """
    Create Walker2D environment with optional POMDP wrapper.
    
    Args:
        env_id: Gymnasium environment ID
        pomdp_type: None (MDP), 'RV', 'FLK', 'RN', 'RSM'
        seed: Random seed
        render_mode: 'human' or 'rgb_array' or None
    
    Returns:
        Wrapped gymnasium environment
    """
    env = gym.make(env_id, render_mode=render_mode)
    
    if pomdp_type == "RV":
        env = RemoveVelocityWrapper(env)
    elif pomdp_type == "FLK":
        env = FlickeringWrapper(env, p_flk=0.2)
    elif pomdp_type == "RN":
        env = RandomNoiseWrapper(env, sigma=0.1)
    elif pomdp_type == "RSM":
        env = RandomSensorMissingWrapper(env, p_rsm=0.1)
    elif pomdp_type is not None:
        raise ValueError(f"Unknown POMDP type: {pomdp_type}. Choose from: RV, FLK, RN, RSM")
    
    env.reset(seed=seed)
    return env


def make_env_fn(env_id="Walker2d-v4", pomdp_type=None, seed=0):
    """Factory function for vectorized environments."""
    def _init():
        return make_env(env_id, pomdp_type, seed)
    return _init
