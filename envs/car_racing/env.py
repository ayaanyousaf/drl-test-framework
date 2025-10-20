import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor

class CarRacingWrapper(gym.Wrapper): 
    """
    Wrapper to prevent Box2D TypeError on windows.
    It converts SB3 float32 actions into Python floats.
    """
    def step(self, action): 
        action = np.array(action, dtype=np.float64)
        return self.env.step(action)

def make_car_racing_env(render_mode=None, seed: int = 7, persona: str = "speedrunner"):
    """
    CarRacing-v2 Game env provided by Gymnasium.
    """
    env = gym.make("CarRacing-v2", render_mode=render_mode, continuous=True)
    env.reset(seed=seed)
    env = CarRacingWrapper(env)
    
    return env