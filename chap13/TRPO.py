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
gamma = 0.9    # decay
tau = 0.01    # lr
delta = 0.01    # conjuagate gradient threshold
buffer_size = int(10e4)
batch_size = 100





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
    

    def log_prob(self, action, mean, log_std):
        # constant term is omitted
        return -log_std - (action-mean)**2 / (2*torch.exp(log_std)**2)


    def grad_log_prob(self, action, mean, log_std):
        log_prob = self.log_prob(action, mean, log_std)
        return autograd.grad(log_prob, list(self.policy.parameters()))
    

    ## g will be obtained from grad_log_prob + advantages at outside of the class


    def get_H(self, kl_div):
        '''
        kl_div : output of self.KL_div()
        '''
        d_kl = autograd.grad(kl_div, list(self.policy.parameters()), create_graph=True)
        return autograd.grad(d_kl, list(self.policy.parameters()))


    def conjugate_gradient(self, H : Tensor, g : Tensor) -> Tensor:    # https://en.wikipedia.org/wiki/Conjugate_gradient_method
        basis = [g]    # orthogonal basis w.r.t inner product H
        s = (g@g)/(g@H@g) * g
        remainder = g - H@s    # error : g-Hs

        while (torch.norm(remainder) > delta):
            base = remainder - torch.sum(torch.Tensor([(p@H@remainder)/(p@H@p)*p for p in basis]))
            basis.append(base)
            s += (base@remainder)/(base@H@base) * base
            remainder = g - H@s

        return s
        

    def forward(self, state : Tensor) -> Tensor:
        value = self.value(state)
        policy = self.policy(state)
        mean, log_std = policy[:action_shape], policy[action_shape:]

        return value, mean, log_std
