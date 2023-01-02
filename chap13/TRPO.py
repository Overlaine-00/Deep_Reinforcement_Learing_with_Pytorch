import numpy as np
import gym

import torch
from torch import tensor
from torch import nn, optim, Tensor, autograd
import torch.nn.functional as F

cpu = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"



### create environment
env = gym.make("Pendulum-v0").unwrapped

state_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]
action_bound = [env.action_space.low, env.action_space.high]


# variables
gamma = 0.9    # decay
tau = 0.01    # lr
delta = 0.01    # restrict condition of TRPO
epsilon = 0.01    # conjuagate gradient threshold

alpha = 1    # backtracking coefficient (update rate of policy network, 0~1. Theoretically alpha=1)
lr = 0.001    # learning rate of value network





### TRPO

class TRPO(nn.Module):
    def __init__(self, state_shape = state_shape, action_shape = action_shape, action_bound = action_bound):
        super().__init__()

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
    

    def log_prob(self, action : Tensor, mean : Tensor, log_std : Tensor) -> Tensor:
        # constant term is omitted
        return -log_std - (action-mean)**2 / (2*torch.exp(log_std)**2)


    def grad_log_prob(self, action : Tensor, mean : Tensor, log_std : Tensor) -> Tensor:
        log_prob = self.log_prob(action, mean, log_std)
        return autograd.grad(log_prob, list(self.policy.parameters()))
    

    ## g will be obtained from grad_log_prob + advantages at outside of the class


    def get_H(self, mean : Tensor, log_std : Tensor) -> Tensor:
        '''
        mean, log_std comes from self.forward()

        Expression : 
        Let \mu = mean, \sigma = variance. Subscript _0 means input(fixed) values.
        We use numpy symbol : *, / = Hadamard(elementwise) or broadcasting operation,
                              @ = matrix multiplication.
        \nabla D_KL = \sigma^{-1} * (\mu-\mu_0) @ \frac{d\mu}{d\phi} + (\sigma^{-2}*\sigma_0 + \sigma^{-1})@\frac{d\mu}{d\phi}
        \nabla^2 D_KL = ( \sigma_0^{-1}*(\frac{d\mu}{d\phi})^\top ) @ \frac{d\mu}{d\phi}
                        - 3( \sigma_0^{-1}*(\frac{d\mu}{d\phi})^\top) ) @ \frac{d\mu}{d\phi}
                        + 2\sigma_0^{-1} @ \frac{d\mu^2}{d^2\phi}
        Here, the term \sigma^{-2}(\mu - \mu_0)**2 \frac{d\mu^2}{d^2\phi} in \nabla D_KL vanishes because
        after substituting \mu=\mu_0, its second derivative becomes zero
        '''
        variance = torch.exp(2*log_std)

        d_mean = autograd.grad(mean, list(self.policy.parameters()))
        d_variance = autograd.grad(variance, list(self.policy.parameters()), create_graph=True)
        d2_variance = autograd.grad(d_variance, list(self.policy.parameters()))

        return (d_mean/variance).T @ d_mean - 3*(d_variance/variance).T @ d_variance + 2*(1/variance) @ d2_variance


    def conjugate_gradient(self, H : Tensor, g : Tensor) -> Tensor:    # https://en.wikipedia.org/wiki/Conjugate_gradient_method
        basis = [g]    # orthogonal basis w.r.t inner product H
        s = (g@g)/(g@H@g) * g
        remainder = g - H@s    # error : g-Hs

        while (torch.norm(remainder) > epsilon):
            base = remainder - torch.sum(torch.Tensor([(p@H@remainder)/(p@H@p)*p for p in basis]))
            basis.append(base)
            s += (base@remainder)/(base@H@base) * base
            remainder = g - H@s

        return s
        

    def forward(self, state : Tensor) -> tuple(Tensor):
        value = self.value(state)
        policy = self.policy(state)
        mean, log_std = policy[:action_shape], policy[action_shape:]

        return value, mean, log_std





### used variables and functions
trpo = TRPO().to(device=device)
optimizer = optim.Adam(trpo.value.parameters(), lr=0.001)


def policy_update(update_amount : Tensor, network = trpo):
    for param, added in zip(network.parameters(), update_amount):
        param.data += added





### training step
num_episodes = 500
num_timesteps = 500

print("Chapter 13.    TRPO")

for i in range(1, num_episodes+1):
    state = env.reset()
    ep_reward = []
    ep_grad_log_prob = []
    ep_value = []

    H = 0

    for t in range(num_timesteps):
        value, mean, log_std = trpo(Tensor(state).to(device=device))
        action = torch.normal(mean, torch.exp(log_std)).item()
        next_state, reward, done, info = env.step(action)

        grad_log_prob = trpo.grad_log_prob(action, mean, log_std).to(device=cpu).numpy()
        value = value.to(device=cpu).numpy()

        ep_reward.append(reward*(gamma**t))    # discounted reward
        ep_grad_log_prob.append(grad_log_prob)
        ep_value.append(value)
        H += trpo.get_H(mean, log_std).to(device=cpu).numpy()

        if done: break
    
    ep_reward = np.array([sum(ep_reward[k:])/(gamma**k) for k in range(ep_reward)])
    ep_grad_log_prob = np.array(ep_grad_log_prob)
    ep_value = np.array(ep_value)

    g = np.sum(ep_grad_log_prob*(ep_reward-ep_value))
    H /= t+1
    s = trpo.conjugate_gradient(H,g)

    policy_update(alpha*torch.sqrt())
