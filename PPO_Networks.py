import torch
from torch import nn


device = "cuda" if torch.cuda.is_available() else "cpu"

class ActorNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)
    def get_log_probs(self, states, actions):
        action_probs = self.forward(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        return log_probs

class CriticNetwork(nn.Module):
    def __init__(self, state_dim=8, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)



if __name__ == "__main__":
    state_dim = 8  # Example state dimension
    action_dim = 2  # Example action dimension
    actor = ActorNetwork(state_dim, action_dim)
    input = torch.tensor([[1,2,3,4,5,6,7,8],
                          [4,3,3,5,2,6,7,8],
                          [1,1,1,1,1,1,1,1]], dtype=torch.float32)
    output = actor(input)
    critic = CriticNetwork(state_dim)
    print(input)
    print(output.shape)