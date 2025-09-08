import gymnasium
from stable_baselines3 import PPO
import MoonJump_env

def load_and_run(agent_name):
    models_dir = f"models/{agent_name}"
    model_path = f"{models_dir}/3950000.zip"

    env = gymnasium.make("MoonJumpEnv-v0", render_mode="human")
    env.reset()

    model = PPO.load(model_path, env=env)
    episodes = 10
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
