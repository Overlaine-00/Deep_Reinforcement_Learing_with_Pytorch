import numpy as np
import gym

import torch
from torch import tensor
from torch import nn, optim, Tensor, tensor, autograd
import torch.nn.functional as F

cpu = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"



### create environment
env = gym.make("Pendulum-v1", g=9.81).unwrapped

state_shape = env.observation_space.shape[0]    # 3
action_shape = env.action_space.shape[0]    # 1
action_bound = np.concatenate([env.action_space.low, env.action_space.high], dtype=np.float32)    # [-2,2]


# variables
epsilon = 0.2
gamma = 0.9    # decay
lr = 0.001
buffer_size = int(10e4)
batch_size = 32





### PPO-clipped
class Policy(nn.Module):
    def __init__(self, state_shape = state_shape, action_shape = action_shape):
        super().__init__()

        self.mutual_network = nn.Sequential(nn.Linear(state_shape, 100, dtype=torch.float), nn.ReLU())
        self.mean = 2*nn.Sequential(nn.Linear(100, action_shape, dtype=torch.float), nn.Tanh())    # 2* : action bound = [-2,2] whereas tanh(x) \in (-1,1)
        self.log_std = nn.Linear(100, action_shape, dtype=torch.float)
    

    def log_prob(self, action : Tensor, mean : Tensor, log_std : Tensor) -> Tensor:
        return -log_std - (action-mean)**2 / (2*torch.exp(log_std)**2)
    

    def forward(self, state : Tensor) -> tuple[Tensor]:
        out = self.mutual_network(state)
        mean = self.mean(out)
        log_std = self.log_std(out)

        return mean, log_std


class Value(nn.Module):
    def __init__(self, state_shape = state_shape):
        super().__init__()

        self.value = nn.Sequential(nn.Linear(state_shape, 100, dtype=torch.float), nn.ReLU(),
                                   nn.Linear(100, 1, dtype=torch.float))
    

    def forward(self, state : Tensor) -> Tensor:
        return self.value(state)






## used variables
policy = Policy().to(deivce=device)
old_policy = Policy().to(deivce=device)
value = Value().to(deivce=device)

value_optimizer = optim.Adam(value.parameters(), lr=lr)

for old_param, param in zip(old_policy.parameters(), policy.parameters()):
    old_param.data.copy_(param.data)


def objective_fn(ratio : Tensor, advantage : Tensor) -> Tensor:
        unclipped = torch.mean(ratio*advantage)
        clipped = torch.mean(torch.clip(ratio, 1-epsilon, 1+epsilon)*advantage)
        return torch.min(unclipped, clipped)





### training
num_episodes = 1000
num_timesteps = 200

print("Chapter 13.    PPO")

for i in range(num_episodes):
    state = env.reset()

    for t in range(num_timesteps):
        mean, log_std = policy(tensor(state).to(devie=device))
        std = torch.exp(std)

        action = torch.normal(mean, std).clamp(action_bound[0], action_bound[1]).requires_grad_(False)
