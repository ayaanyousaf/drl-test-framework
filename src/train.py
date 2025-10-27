import argparse
import os
import yaml
import gymnasium as gym

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.lunar_lander.env import LunarLanderEnv
from envs.swaglabs.env import SwagLabsEnv


def make_env(app="lunar_lander", persona="baseline", render_mode=None, seed=7):
    """
    Function to build an instance of the app env.
    Applies Monitor SB3 wrapper for logging episode stats.
    """
    if (app == "lunar_lander"): 
        app_name = "lunar"
        env = LunarLanderEnv(persona=persona, render_mode=render_mode)

    elif app == "swaglabs": 
        app_name = "swaglabs"
        env = SwagLabsEnv(persona=persona)

    else:
        raise ValueError(f"App does not exist: {app}")
    
    env = Monitor(env, filename=f"logs/{app}/{app_name}_{seed}.monitor.csv")

    return env

def load_hyperparams(algo, app):
    """
    Loads hyperparameters based on app and algorithm.
    Loads YAML files found in configs for reusability.
    """

    path = f"configs/algo/{algo}.yaml"

    with open(path, "r") as file:
        config = yaml.safe_load(file)

    if app in config: 
        return config[app]

    return config.get("default")

def main(): 
    # Create command line arguments using argparse
    p = argparse.ArgumentParser()
    p.add_argument("--app", choices=["lunar_lander","swaglabs"], default="lunar_lander")
    p.add_argument("--algo", choices=["ppo","a2c"], default="ppo")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--persona", choices=["baseline", "speedrunner", "safe", "functional", "explorer"], default="baseline")
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--model_dir", default="models")

    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Make vectorized env for SB3
    vec_env = DummyVecEnv([lambda: make_env(app=args.app, persona=args.persona, render_mode=None, seed=args.seed)])

    # Pick algorithm (PPO vs. A2C)
    if args.algo == "ppo": 
        Algo = PPO
    else: 
        Algo = A2C

    policy = "MlpPolicy"

    model = Algo(
        policy,
        vec_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.log_dir,
        **load_hyperparams(args.algo, args.app),
    )

    # Store short app name for path
    app_name = "lunar" if args.app == "lunar_lander" else "swaglabs"

    # Build clean tensorboard log directories
    log_app_dir = os.path.join(args.log_dir, args.app)
    os.makedirs(log_app_dir, exist_ok=True)
    log_dir = os.path.join(log_app_dir, f"{app_name}_{args.algo}_{args.persona}_{args.timesteps}") 
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    # Build clean model directories
    model_app_dir = os.path.join(args.model_dir, args.app)
    os.makedirs(model_app_dir, exist_ok=True)
    path = os.path.join(model_app_dir, f"{app_name}_{args.algo}_{args.persona}_{args.timesteps}.zip")
    model.save(path)
    print("Saved: ", path)

if __name__ == "__main__":
    main()
