import torch

class PPOMemory:
    def __init__(self, rollout_len, state_dim, device):
        self.device = device
        self.state_dim = state_dim
        self.rollout_len = rollout_len
        self.states = torch.zeros((rollout_len, state_dim), device=device)
        self.actions = torch.zeros(rollout_len, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(rollout_len, device=device)
        self.rewards = torch.zeros(rollout_len, device=device)
        self.dones = torch.zeros(rollout_len, device=device)
        self.values = torch.zeros(rollout_len, device=device)

        self.returns = torch.zeros(rollout_len, device=device)
        self.advantages = torch.zeros(rollout_len, device=device)
        self.ptr = 0

    def store(self, state, action, log_prob, reward, done, value):
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float32)
        self.actions[self.ptr] = torch.as_tensor(action)
        self.log_probs[self.ptr] = torch.as_tensor(log_prob.detach())
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.float32)
        self.values[self.ptr] = torch.as_tensor(value.detach(), dtype=torch.float32)
        self.ptr += 1

    def compute_returns_advantages(self, last_value, discount_factor=0.99, GAE_lambda=0.95):
        with torch.no_grad():
            gae = 0
            next_value = last_value
            for t in reversed(range(self.rollout_len)):
                mask = 1 - self.dones[t]
                temporal_diff_error = self.rewards[t] + discount_factor * next_value * mask - self.values[t]
                gae = temporal_diff_error + discount_factor * GAE_lambda * gae * mask
                self.advantages[t] = gae
                self.returns[t] = self.advantages[t] + self.values[t]
                next_value = self.values[t]
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std(unbiased=False) + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std
        print("advantages:", self.advantages)

    def clear(self):
        self.ptr = 0
        self.states = torch.zeros((self.rollout_len, self.state_dim), device=self.device)
        self.actions = torch.zeros(self.rollout_len, dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros(self.rollout_len, device=self.device)
        self.rewards = torch.zeros(self.rollout_len, device=self.device)
        self.dones = torch.zeros(self.rollout_len, device=self.device)
        self.values = torch.zeros(self.rollout_len, device=self.device)
        self.returns = torch.zeros(self.rollout_len, device=self.device)
        self.advantages = torch.zeros(self.rollout_len, device=self.device)

    def is_full(self):
        return self.ptr == self.rollout_len


if __name__ == "__main__":
    mem = PPOMemory(5, 8)
    print(mem.states)
    mem.store([1,2,3,4,5,6,7,8], 1, 0.5, 1, 0, 0.8)
    print(mem.states)