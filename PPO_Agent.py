from collections import deque
import torch
from torch import nn
from PPO_Memory import PPOMemory
from  PPO_Networks import ActorNetwork, CriticNetwork
import gymnasium as gym
import MoonJump_env
import yaml
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
from datetime import datetime, timedelta
RUNS_DIR = "models"
os.makedirs(RUNS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class PPOAgent:
    def __init__(self, agent_name):
        with open("hyperparameters.yml", 'r') as file:
            self.config = yaml.safe_load(file)
            hyperparams = self.config["Agent1"]        
        self.lr = hyperparams['lr']
        self.gamma = hyperparams['gamma']
        self.rollout_len = hyperparams['rollout_len']
        self.minibatch_size = hyperparams['minibatch_size']
        self.eps = hyperparams['eps']
        self.update_epochs = hyperparams['update_epochs']
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{agent_name}.pth")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{agent_name}_graph.png")
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{agent_name}.log")
    def run(self, is_training=True):
        last_graph_update = datetime.now()
        env = gym.make("MoonJumpEnv-v0", render_mode="human" if not is_training else None)
        num_states = env.observation_space.shape[0]
        actor = ActorNetwork(state_dim=num_states).to(device)
        if is_training:
            critic = CriticNetwork(state_dim=num_states).to(device)
            memory = PPOMemory(self.rollout_len, num_states, device)
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.lr)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.lr)
            best_reward = -99999

            stats_window_size = 100
            ep_len_buffer = deque(maxlen=stats_window_size)
            ep_rew_buffer = deque(maxlen=stats_window_size)

            rollout_avg_rewards =[]
            rollout_avg_lengths = []
        else:
            actor.load_state_dict(torch.load(self.MODEL_FILE))
            actor.eval()

        prev_rollout = rollout = 0
        for episodes in itertools.count():
            state, info = env.reset()
            done = False
            ep_reward = 0
            ep_len = 0
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            while not done:
                action_probs = actor(state)
                if is_training:
                    dist = torch.distributions.Categorical(action_probs)
                    action = dist.sample()
                    log_probs = dist.log_prob(action)
                    value_estimate = critic(state).squeeze()
                    next_state, reward, terminated, truncated, info = env.step(action.item())
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                    done = terminated or truncated
                    ep_reward += reward
                    ep_len += 1
                    memory.store(state, action, log_probs, reward, done, value_estimate)
                    if memory.is_full():
                        last_value = critic(next_state).squeeze().item() if done == False else 0.0
                        memory.compute_returns_advantages(last_value=last_value, discount_factor=self.gamma)
                        self.optimize(actor, critic, memory, actor_optimizer, critic_optimizer)
                        memory.clear()
                        rollout += 1
                    
                else:
                    action = action_probs.argmax().item()
                    next_state, reward, terminated, truncated, info = env.step(action)
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                    done = terminated or truncated
                state = next_state


            if is_training:
                ep_len_buffer.append(ep_len)
                ep_rew_buffer.append(ep_reward)
                if prev_rollout!= rollout:
                    rollout_avg_rewards.append(np.mean(ep_rew_buffer))
                    rollout_avg_lengths.append(np.mean(ep_len_buffer))
                    self.save_graph(rollout_avg_rewards, rollout_avg_lengths)
                    prev_rollout = rollout
                if ep_reward > best_reward:
                    best_reward = ep_reward
                    torch.save(actor.state_dict(), self.MODEL_FILE)
                    log_msg = f"Episode: {episodes}, New best reward: {best_reward}"
                    print(log_msg)
                    with open(self.LOG_FILE, 'a') as log_file:
                        log_file.write(f"{log_msg}\n")

    def optimize(self, actor: ActorNetwork, critic: CriticNetwork, memory: PPOMemory, actor_optimizer: torch.optim.Adam,
                  critic_optimizer: torch.optim.Adam):
        for epoch in range(self.update_epochs):  
            perm = torch.randperm(self.rollout_len)
            for batch_idx in range(0, self.rollout_len, self.minibatch_size):
                batch_slice = perm[batch_idx:batch_idx + self.minibatch_size]
                b_states      = memory.states[batch_slice]
                b_actions     = memory.actions[batch_slice]
                b_returns     = memory.returns[batch_slice]
                b_advantages  = memory.advantages[batch_slice]
                b_old_log_probs = memory.log_probs[batch_slice]
                
                #Optimize Critic 
                values_pred = critic(b_states).squeeze()
                mse_loss = nn.MSELoss()
                critic_loss = 0.5*mse_loss(values_pred, b_returns)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                #optimize Actor
                log_probs = actor.get_log_probs(b_states, b_actions)
                ratios = torch.exp(log_probs - b_old_log_probs)

                obj1 = ratios * b_advantages
                obj2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * b_advantages
                actor_loss = -torch.min(obj1, obj2).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

    def save_graph(self, rollout_avg_rewards, rollout_avg_lengths):

            fig = plt.figure(1)

            plt.subplot(121)

            plt.ylabel('Mean Rewards per Rollout')
            plt.plot(rollout_avg_rewards)


            plt.subplot(122) 
            plt.ylabel('Mean Episode Lengths per Rollout')
            plt.plot(rollout_avg_lengths)

            plt.subplots_adjust(wspace=1.0, hspace=1.0)

            fig.savefig(self.GRAPH_FILE)
            plt.close(fig)

if __name__ == "__main__":
    agent = PPOAgent("Custom_ppo")
    while True:
        agent.run(is_training=False)