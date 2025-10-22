import argparse
import os
import gymnasium as gym

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from envs.car_racing.env import CarRacingEnv
# from envs.erpnext.env import ERPNextEnv (when ready)


def make_env(app="car_racing", persona="baseline", render_mode=None, seed=7):
    """
    Function to build an instance of the app env.
    Applies Monitor SB3 wrapper for logging episode stats.
    """
    if (app == "car_racing"): 
        app_name = "car"
        env = CarRacingEnv(persona=persona, render_mode=render_mode, seed=seed)

    # TODO: Make env for ERPNext once it's ready
    elif app == "erpnext": 
        app_name = "erp"
        env = ...
    
    env = Monitor(env, filename=f"logs/{app_name}_{seed}.monitor.csv")

    return env

def get_hyperparams(algo, app):
    """
    Select hyperparameters based on app and algorithm.
    Makes sure the hyperparameters make sense for your app.
    """
    if algo == "ppo":
        if app == "car_racing":
            # For envs with visuals (CarRacing-v2)
            return dict(
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.001,
            )
        else:
            # For envs with no visuals (ERPNext)
            return dict(
                learning_rate=3e-4,
                n_steps=1024,
                batch_size=256,
                n_epochs=10,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.001,
            )

    elif algo == "a2c":
        # A2C hyperparameters (same for all envs)
        return dict(
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            ent_coef=0.0001,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

def main(): 
    # Create command line arguments using argparse
    p = argparse.ArgumentParser()
    p.add_argument("--app", choices=["car_racing","erpnext"], default="car_racing")
    p.add_argument("--algo", choices=["ppo","a2c"], default="ppo")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--persona", choices=["baseline", "speedrunner", "safe"], default="baseline")
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--model_dir", default="models")

    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # Make vectorized env for SB3
    vec_env = DummyVecEnv([lambda: make_env(app=args.app, persona=args.persona, render_mode=None)])
    vec_env.seed(args.seed)

    if args.app == "car_racing": 
        vec_env = VecTransposeImage(vec_env) # convert HWC to CHW for CNN policy (image-based envs)

    # Pick algorithm (PPO vs. A2C)
    if args.algo == "ppo": 
        Algo = PPO
    else: 
        Algo = A2C

    policy = "CnnPolicy" if args.app == "car_racing" else "MlpPolicy"
    model = Algo(
        policy,
        vec_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.log_dir,
        **get_hyperparams(args.algo, args.app),
    )

    # Store short app name for path
    app_name = "car" if args.app == "car_racing" else "erp"

    # Build clean tensorboard log directories
    log_app_dir = os.path.join(args.log_dir, args.app)
    os.makedirs(log_app_dir, exist_ok=True)
    log_dir = os.path.join(log_app_dir, f"{app_name}_{args.algo}_{args.persona}_seed{args.seed}") 
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    # Build clean model directories
    model_app_dir = os.path.join(args.model_dir, args.app)
    os.makedirs(model_app_dir, exist_ok=True)
    path = os.path.join(model_app_dir, f"{app_name}_{args.algo}_{args.persona}_seed{args.seed}.zip")
    model.save(path)
    print("Saved: ", path)

if __name__ == "__main__":
    main()
