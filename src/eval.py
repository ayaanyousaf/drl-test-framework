import argparse
import numpy as np
from stable_baselines3 import PPO, A2C
from envs.lunar_lander.env import LunarLanderEnv
from envs.swaglabs.env import SwagLabsEnv
from .export import export_metrics_csv

def evaluate_swaglabs(model, env, episodes=5):
    """
    Evaluate a trained model on the Swag Labs environment.
    """

    rewards, successes, errors, episode_metrics = [], [], [], []

    for ep in range(episodes):
        obs, info = env.reset()
        terminated, truncated = False, False
        episode_rewards = []
        total_success = 0
        total_error = 0
        steps = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)

            steps += 1  # increment for each step

            total_success += info.get("success", 0)
            total_error += info.get("error", 0)
        
        total_reward = np.sum(episode_rewards)
        rewards.append(total_reward)
        successes.append(total_success)
        errors.append(total_error)

        episode_metrics.append({
            "episode": ep + 1,
            "total_reward": float(total_reward),
            "total_success": int(total_success),
            "total_error": int(total_error),
            "steps": steps,
        })

        print(f"Episode {ep+1}: reward={total_reward:.2f}, "f"success={total_success}, error={total_error}, steps={steps}")

    results = {
        "avg_reward": np.mean(rewards),
        "avg_success": np.mean(successes),
        "avg_error": np.mean(errors),
    }

    return results, episode_metrics
            

def evaluate_lunar(model, env, episodes=5):
    """
    Evaluate a trained model on the Lunar Lander environment.
    """
    rewards, crashes, landings, episode_metrics = [], [], [], []

    for ep in range(episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        episode_rewards = []
        crashed, landed = False, False
        landing_type = None
        steps = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)

            steps += 1 # increment for each step

            # Get crash/landings from env wrapper
            crashed = crashed or info.get("crashed", False)
            landed = landed or info.get("landed", False)

            if info.get("landing_type"):
                landing_type = info["landing_type"] 

        landing_time = steps if landed else None
        total_reward = np.sum(episode_rewards)
        rewards.append(total_reward)
        crashes.append(int(crashed))
        landings.append(int(landed))

        # Add current episodes data to metrics list to export later
        episode_metrics.append({
            "episode": ep + 1,
            "total_reward": float(np.sum(episode_rewards)),
            "landing_type": landing_type,
            "crashed": crashed, 
            "landed": landed,
            "landing_time": landing_time,
        })

        print(f"Episode {ep+1}: reward={total_reward:.2f}, "f"landed={landed}, crashed={crashed}, landing_type={landing_type}, landing_time={landing_time}")

    results = {
        "avg_reward": np.mean(rewards),
        "crash_rate": np.mean(crashes),
        "landing_rate": np.mean(landings),
    }
        
    return results, episode_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app", choices=["lunar_lander", "swaglabs"], default="lunar_lander")
    p.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    p.add_argument("--persona", choices=["baseline", "speedrunner", "safe", "functional", "explorer"], default="baseline")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--render", action="store_true")
    p.add_argument("--export", action="store_true")
    args = p.parse_args()

    # Choose the algorithm
    algo = PPO if args.algo == "ppo" else A2C

    app_name = "lunar" if args.app == "lunar_lander" else "swaglabs"
    file_name = f"{app_name}_{args.algo}_{args.persona}_{args.timesteps}"

    # Create model path
    model_path = f"models/{args.app}/{file_name}.zip"

    # Load model
    model = algo.load(model_path)
    print(f"Loaded model: {model_path}")

    render_mode = "human" if args.render else None

    # Create correct env and evaluate based on app
    if args.app == "lunar_lander":
        env = LunarLanderEnv(persona=args.persona, render_mode=render_mode, seed=7)
        results, episode_metrics = evaluate_lunar(model, env, episodes=args.episodes)

        print(f"\n--- Evaluation Results ({args.algo.upper()} | {args.persona}) ---")
        print(f"Average Reward: {results['avg_reward']:.2f}")
        print(f"Landing Rate: {results['landing_rate']*100:.2f}%")
        print(f"Crash Rate: {results['crash_rate']*100:.2f}%\n")
    
    else: 
        env = SwagLabsEnv(persona=args.persona)
        results, episode_metrics = evaluate_swaglabs(model, env, episodes=args.episodes)

        print(f"\n--- Evaluation Results ({args.algo.upper()} | {args.persona}) ---")
        print(f"Average Reward: {results['avg_reward']:.2f}")
        print(f"Average Success: {results['avg_success']:.2f}")
        print(f"Average Errors: {results['avg_error']:.2f}\n")

    # Evaluate
    env.close()

    # Export per-episode metrics to CSV
    export_dir = f"logs/{args.app}/{file_name}"

    if args.export: 
        export_metrics_csv(episode_metrics, export_dir=export_dir)
    else: 
        print("Metrics not exported. Use --export flag to export metrics to CSV.")


if __name__ == "__main__":
    main()
