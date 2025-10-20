import argparse
import os
import gymnasium as gym

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from envs.car_racing.env import make_car_racing_env

def make_env(render_mode=None, seed=7):
    """
    Function to build an instance of CarRacing env.
    Applies Monitor SB3 wrapper for logging episode stats.
    """
    env = make_car_racing_env(render_mode=render_mode, seed=seed)
    env = Monitor(env, filename=f"logs/car_{seed}.monitor.csv")

    return env

def main(): 
    # Create command line arguments using argparse
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo","a2c"], default="ppo")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--persona", choices=["speedrunner","safedriver"], default="speedrunner")
    p.add_argument("--save_dir", default="models")

    args = p.parse_args()

    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # Make vectorized env for SB3 
    vec_env = DummyVecEnv([lambda: make_env(render_mode=None, seed=args.seed)])
    vec_env = VecTransposeImage(vec_env) # convert HWC to CHW for CNN policy

    # Pick algorithm (PPO vs. A2C)
    if args.algo == "ppo": 
        Algo = PPO
    else: 
        Algo = A2C

    model = Algo("CnnPolicy", vec_env, verbose=1, seed=args.seed, tensorboard_log="logs/tb")

    #  For clean tensorboard folders
    tb_dir = os.path.join("logs", "tb")
    os.makedirs(tb_dir, exist_ok=True)
    log_dir = os.path.join(tb_dir, f"{args.algo}_{args.persona}_seed{args.seed}") 

    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    out = os.path.join(args.save_dir, f"car_{args.algo}_{args.persona}_seed{args.seed}.zip")
    model.save(out)
    print("Saved: ", out)

if __name__ == "__main__":
    main()
