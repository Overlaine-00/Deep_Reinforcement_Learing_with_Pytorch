import numpy as np
import gym

import torch
from torch import tensor
from torch import nn, optim, Tensor, tensor, autograd
import torch.nn.functional as F

cpu = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Chapter 14.    Categorical DQN")



### create environment
env = gym.make("Tennis-v4").unwrapped

state_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]

# variables
v_min, v_max = 0, 1000
atoms = 51
gamma = 0.99    # discount factor
batch_size = 64
update_target_network = 50    # target network update period
epsilon = 0.5    # epsilon-greedy policy



### categorical DQN class
class categorical_DQN(nn.Module):
    def __init__(self, state_shape = state_shape, action_shape = action_shape,
                       v_min = v_min, v_max = v_max, atoms = atoms):
        super().__init__()
        self.state_shape, self.action_shape = state_shape, action_shape
        self.v_min, self.v_max = v_min, v_max
        self.atoms = atoms

        self.value_gap = (v_max-v_min)/atoms
        self.value_list = [v_min + i*self.value_gap for i in range(atoms)]

        self.network1 = nn.Sequential(nn.Conv2d(3, 6, 5), nn.ReLU(),
                                      nn.Conv2d(6, 12, 3), nn.ReLU(),
                                      nn.Flatten(),
                                      nn.LazyLinear(24), nn.Tanh(),
                                      nn.Linear(24,24), nn.Tanh())

        self.network2 = nn.Sequential(nn.Linear(24+action_shape, atoms), nn.Softmax())
    
    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        out = self.network1(state)
        out = torch.cat([out, action], dim=0)
        out = self.network2(out)
        return out
