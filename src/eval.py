import argparse
import numpy as np
from stable_baselines3 import PPO, A2C
from envs.lunar_lander.env import LunarLanderEnv
from .metrics import export_metrics_csv


def evaluate(model, env, episodes=5):
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
    p.add_argument("--app", choices=["lunar_lander", "erpnext"], default="lunar_lander")
    p.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    p.add_argument("--persona", choices=["baseline", "speedrunner", "safe"], default="baseline")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    # Choose the algorithm
    algo = PPO if args.algo == "ppo" else A2C

    app_name = "lunar" if args.app == "lunar_lander" else "erp"
    file_name = f"{app_name}_{args.algo}_{args.persona}_{args.timesteps}"

    # Create model path
    model_path = f"models/{args.app}/{file_name}.zip"

    # Load model
    model = algo.load(model_path)
    print(f"Loaded model: {model_path}")

    # Create environment
    render_mode = "human" if args.render else None
    env = LunarLanderEnv(persona=args.persona, render_mode=render_mode, seed=7)

    # Evaluate
    results, episode_metrics = evaluate(model, env, episodes=args.episodes)
    env.close()

    print(f"\n--- Evaluation Results ({args.algo.upper()} | {args.persona}) ---")
    print(f"Average Reward: {results['avg_reward']:.2f}")
    print(f"Landing Rate: {results['landing_rate']*100:.2f}%")
    print(f"Crash Rate: {results['crash_rate']*100:.2f}%\n")

    # Export per-episode metrics to CSV
    export_dir = f"logs/{args.app}/{file_name}"
    export_metrics_csv(episode_metrics, export_dir=export_dir)


if __name__ == "__main__":
    main()
