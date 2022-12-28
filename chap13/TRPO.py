import numpy as np
import gym

import torch
from torch import tensor
from torch import nn, optim, Tensor, autograd
import torch.nn.functional as F



### create environment
env = gym.make("Pendulum-v0").unwrapped

state_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]
action_bound = [env.action_space.low, env.action_space.high]


# variables
delta = 0.01





class TRPO:
    def __init__(self, state_shape = state_shape, action_shape = action_shape, action_bound = action_bound):
        self.action_shape = action_shape
        self.low, self.high = action_bound

        # value network
        self.value = nn.Sequential(nn.Linear(state_shape, 400, dtype=torch.float),
                                   nn.ReLU(),
                                   nn.Linear(400, 300, dtype=torch.float),
                                   nn.ReLU(),
                                   nn.Linear(300, 1, dtype=torch.float))
        
        # policy network : assumed to be normal. Returns mean and log-std
        self.policy = nn.Sequential(nn.Linear(state_shape, 300, dtype=torch.float),
                                    nn.ReLU(),
                                    nn.Linear(300, 200, dtype=torch.float),
                                    nn.ReLU(),
                                    nn.Linear(200, 2*action_shape, dtype=torch.float))
    
    def KL_div(self, old_mean, old_log_sigma, new_mean, new_log_sigma):
        '''
        old_theta, new_theta : output of self.policy
        '''
        old_sigma, new_sigma = torch.exp(old_log_sigma), torch.exp(new_log_sigma)

        return torch.sum( old_sigma/new_sigma ) + \
               torch.sum( ((new_mean-old_mean)**2)/new_sigma ) + \
               torch.log(torch.prod(new_sigma)/torch.prod(old_sigma))


    def get_g(self, L):
        return autograd.grad(L, list(self.policy.parameters()), retain_graph=True)

    def get_H(self, kl_div):
        '''
        kl_div : output of self.KL_div()
        '''
        d_kl = autograd.grad(kl_div, list(self.policy.parameters()), create_graph=True)
        return autograd.grad(d_kl, list(self.policy.parameters()))
    
    def log_prob(self, action, mean, log_std):
        # constant term is omitted
        return -log_std - (action-mean)**2 / (2*torch.exp(log_std)**2)
        
    def forward(self, state : Tensor) -> Tensor:
        value = self.value(state)
        policy = self.policy(state)
        mean, log_std = policy[:action_shape], policy[action_shape:]

        return value, mean, log_std
