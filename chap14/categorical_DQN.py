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
