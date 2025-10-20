import gymnasium as gym
from stable_baselines3 import PPO
from envs.car_racing.env import make_car_racing_env

# Load model
model = PPO.load("models/car_ppo_speedrunner_seed7.zip")

# Create env for evaluation
env = make_car_racing_env(render_mode="human")
obs, _ = env.reset(seed=7)

# Run a few episodes
for _ in range(3):
    done, truncated = False, False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
env.close()
