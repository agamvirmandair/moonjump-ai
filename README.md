MoonJump: A Custom 2D Gymnasium Environment with Proximal Policy Optimization - PPO

📌Project Overview

This project explores reinforcement learning by creating a custom 2D game environment in Gymnasium and training agents using Proximal Policy Optimization (PPO).
The game, MoonJump, challenges an agent-controlled robot to dodge incoming obstacles of varying speeds, sizes, and distances.

Key contributions:
  - A custom Gymnasium environment (MoonJump-v0) built with Pygame.
  - Baseline PPO training using Stable Baselines3.
  - A from-scratch PPO implementation in PyTorch for deeper understanding.
  - Comparison between SB3’s performance and the custom PPO agent.


🎮Environment Design

Action space:
  - Discrete(2) → 0 = do nothing, 1 = jump.

Observation space (vector of floats):
  - Robot’s vertical position
  - Robot’s vertical velocity
  - Distance to the nearest obstacle.
  - Height of nearest obstacle.
  - Obstacle’s horizontal speed.

Reward function:

  reward = 0.1 + score*0.01 rewards for each frame survived 
  reward = +1 reward for every obstacle avoided.
  reward = -5 + (PLAYER_X - self.obstacles_pos[self.incoming_obstacle]) * 0.01


-5 penalty → Applied when the agent collides with an obstacle. This creates a strong negative signal that teaches the agent to avoid crashes.

Progress bonus (PLAYER_X - obstacle_x) * 0.01 → Small positive reward that increases as the player successfully passes obstacles. The farther the agent moves relative to the next obstacle, the higher the reward.

If the agent does nothing and crashes → strong penalty.
If the agent survives longer and clears obstacles → accumulates reward.

This balance encourages long-term survival strategies rather than just random jumping.

Termination:
  - Robot collides with an obstacle.

Training Setup:
  - Algorithms: PPO (baseline + custom).

Libraries:
  - Gymnasium
  - Stable Baselines3
  - PyTorch

Hyperparameters (SB3 PPO):
  - Learning rate: 3e-4
  - γ (discount factor): 0.99
  - n-steps: 2048
  - Clip range: 0.2

Custom PPO highlights:

Implemented advantage estimation, policy clipping, and separate actor-critic networks.

Training loop built from scratch with PyTorch.

📊 Results:

Both agents learned jumping strategies, but SB3 converged faster and was more reilable.
  - SB3 PPO: Best average score ≈ 500 (max_len).
    More stable & sample efficient
    
  - Custom PPO: Best average score ≈ 100.
    Slower, but shows correct learning dynamics



(Insert training curve plots here: rewards vs. timesteps)

🎥 Demo

(Insert a GIF or YouTube link of the trained agent playing)


Optimize observation encoding for faster training.
