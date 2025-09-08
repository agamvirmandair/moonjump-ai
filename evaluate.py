import load_agent_stable
import argparse
from PPO_Agent import PPOAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=["sb3", "custom"], required=True, help="Choose which agent to evaluate: sb3 or custom")
    args = parser.parse_args()
    if args.agent == "sb3":
        load_agent_stable.load_and_run("sb3_ppo")
    elif args.agent == "custom":
        agent = PPOAgent("Custom_ppo")
        for x in range(10):
            agent.run(is_training=False)