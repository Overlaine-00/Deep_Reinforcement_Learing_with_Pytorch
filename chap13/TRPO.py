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
delta = 0.01    # conjuagate gradient threshold
buffer_size = int(10e4)
batch_size = 100





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
        

    def forward(self, state : Tensor) -> tuple(Tensor):
        value = self.value(state)
        policy = self.policy(state)
        mean, log_std = policy[:action_shape], policy[action_shape:]

        return value, mean, log_std





### Replay Buffer

state_index = [_ for _ in range(state_shape)]
action_index = [_ + state_shape for _ in range(action_shape)]
reward_index = [state_shape + action_shape]
next_state_index = [_ + state_shape + action_shape + 1 for _ in range(state_shape)]
class Replay_Buffer:
    def __init__(self, capacity : int = buffer_size, state_shape : int = state_shape, action_shape : int = action_shape):
        '''
        basic replay buffer.
        store : state, action, reward, next_state
        '''
        self.capacity = capacity
        self.pointer = 0
        
        self.buffer = np.zeros([capacity, 2*state_shape + action_shape + 1], dtype=np.float32)
    
    def append(self, state : np.ndarray, action : np.float32, reward : np.float32, next_state : np.ndarray):
        index = self.pointer%self.capacity
        self.buffer[index] = np.concatenate([state, np.array([action, reward]), next_state])
        
        self.pointer += 1
        if self.pointer >= 2*self.capacity: self.pointer -= self.capacity
        
        del index
    
    def batch_sample(self, batch_size : int) -> np.ndarray:
        # choose sample by bootstapping
        index = np.random.randint(0, min(self.pointer, self.capacity), (batch_size,))
        
        return self.buffer[index]



### used variables
trpo = TRPO().to(device=device)
buffer = Replay_Buffer()

optimizer = optim.Adam(trpo.parameters(), lr=0.001)




### training step
num_episodes = 500
num_timesteps = 500

print("Chapter 13.    TRPO")

for i in range(num_episodes):
    state = env.reset()
    ep_reward = []
    ep_grad_log_prob = []
    ep_value = []

    for t in range(num_timesteps):
        value, mean, log_std = trpo(Tensor(state).to(device=device))
        action = torch.normal(mean, torch.exp(log_std)).item()
        next_state, reward, done, info = env.step(action)

        value = value.to(device=cpu).item()
        mean = mean.to(device=cpu).item()
        log_std = log_std.to(device=cpu).item()

        ep_reward.append(reward)
        ep_grad_log_prob.append(trpo.grad_log_prob(action, mean, log_std))
        ep_value.append(value)

        if done: break
    
    ep_reward = np.array(ep_reward)
    ep_grad_log_prob = np.array(ep_grad_log_prob)
    ep_value = np.array(ep_value)

    g = np.sum(ep_grad_log_prob*(ep_reward))

