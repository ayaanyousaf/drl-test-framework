import gymnasium as gym
import numpy as np
from .reward import RewardManager

class LunarLanderEnv(gym.Wrapper):
    """
    Wrapper for LunarLander-v2 environment.
    Adds velocity, tilt, and landing/crashed info to RewardManager.
    """

    def __init__(self, persona="baseline", render_mode=None, seed=None):
        env = gym.make("LunarLander-v2", render_mode=render_mode)
        super().__init__(env)

        self.persona = persona
        self.frame_count = 0
        self.landed = False
        self.crashed = False

        self.reward_manager = None if persona == "baseline" else RewardManager(persona)

    def reset(self, **kwargs):
        """
        Resets the env state for each episode.
        Call this method before starting a new episode.

        Return: 
            obs: np.ndarray vector of observations
            info: dict containing episode metrics and physics variables (velocity, position, etc.)
        """
        obs, info = self.env.reset(seed=kwargs.get("seed", None))

        # Reset state
        self.frame_count = 0
        self.landed = False
        self.crashed = False

        if self.reward_manager:
            self.reward_manager.reset()

        return obs, info

    def step(self, action):
        """
        Goes forward a single step in the environment using given action.
        Auto-collects metrics in info dictionary for RewardManager to use.
        Detects crashes and landings using custom and Gym-provided boolean flags.

        Args: 
            action: int or np.ndarray action for the LunarLander agent.

        Return: 
            obs: observation vector of the next state.
            reward: a float reward value (shaped in RewardManager) for the step.
            terminated: a boolean that tracks if an episode has ended.
            truncated: a boolean that tracks if an episode reached its time limit.
            info: contains episode metrics and physics info (velocity, position, etc.)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.frame_count += 1

        # Add metrics to observation
        x, y, x_dot, y_dot, theta, theta_dot, leg1, leg2 = obs

        info["x_pos"] = float(x)
        info["y_pos"] = float(y)
        info["x_vel"] = float(x_dot)
        info["y_vel"] = float(y_dot)
        info["angle"] = float(theta)
        info["angular_vel"] = float(theta_dot)
        info["leg1_contact"] = bool(leg1)
        info["leg2_contact"] = bool(leg2)

        # Landing/crash detection
        between_flags = abs(x) < 0.2
        if leg1 and leg2 and between_flags: # count as landing if it lands between flags and both legs touch the ground
            self.landed = True
            terminated = True
        elif leg1 and leg2 and y < 0.05: 
            self.crashed = True
            terminated = True
        elif leg1 or leg2: 
            self.crashed = True
            terminated = True

        info["landed"] = self.landed
        info["crashed"] = self.crashed

        # Apply custom reward shaping based on persona (overwrites default env reward)
        if self.reward_manager:
            reward = self.reward_manager.compute(info)

        # Add automatic metric collection
        info["frame"] = self.frame_count
        info["persona"] = self.persona
        info["total_reward"] = float(reward)

        return obs, reward, terminated, truncated, info