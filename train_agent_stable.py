import gymnasium
from stable_baselines3 import PPO
import os 
import MoonJump_env
models_dir = "models"
logsdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logsdir):
    os.makedirs(logsdir)

env = gymnasium.make("MoonJumpEnv-v0")
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logsdir)
TIMESTEPS = 10000
for i in range(1,100000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="sb3_ppo")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

