import argparse
import numpy as np
from stable_baselines3 import PPO, A2C
from envs.lunar_lander.env import LunarLanderEnv


def evaluate(model, env, episodes=5):
    rewards, crashes, landings = [], [], []

    for ep in range(episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        episode_rewards = []
        crashed, landed = False, False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_rewards.append(reward)

            # Track crash/landings from env wrapper
            crashed = crashed or info.get("crashed", False)
            landed = landed or info.get("landed", False)

        total_reward = np.sum(episode_rewards)
        rewards.append(total_reward)
        crashes.append(int(crashed))
        landings.append(int(landed))

        print(
            f"Episode {ep+1}: reward={total_reward:.2f}, "
            f"landed={landed}, crashed={crashed}"
        )

    return {
        "avg_reward": np.mean(rewards),
        "crash_rate": np.mean(crashes),
        "landing_rate": np.mean(landings),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app", choices=["lunar_lander", "erpnext"], default="lunar_lander")
    p.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    p.add_argument("--persona", choices=["baseline", "speedrunner", "safe"], default="baseline")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    # Choose the algorithm
    algo = PPO if args.algo == "ppo" else A2C

    # Create model path
    app_name = "lunar" if args.app == "lunar_lander" else "erp"
    path = f"models/{args.app}/{app_name}_{args.algo}_{args.persona}_{args.timesteps}.zip"

    # Load model
    model = algo.load(path)
    print(f"Loaded model: {path}")

    # Create environment
    render_mode = "human" if args.render else None
    env = LunarLanderEnv(persona=args.persona, render_mode=render_mode, seed=7)

    # Evaluate
    results = evaluate(model, env, episodes=args.episodes)
    env.close()

    print(f"\n=== Evaluation Results ({args.algo.upper()} | {args.persona}) ===")
    print(f"Average Reward: {results['avg_reward']:.2f}")
    print(f"Landing Rate: {results['landing_rate']*100:.2f}%")
    print(f"Crash Rate: {results['crash_rate']*100:.2f}%\n")


if __name__ == "__main__":
    main()
