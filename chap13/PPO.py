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
policy_network = Policy().to(deivce=device)
old_policy_network = Policy().to(deivce=device)
value_network = Value().to(deivce=device)

policy_optimizer = optim.Adam(policy_network.parameters(), lr=lr)
value_optimizer = optim.Adam(value_network.parameters(), lr=lr)

for old_param, param in zip(old_policy_network.parameters(), policy_network.parameters()):
    old_param.data.copy_(param.data)


def objective_fn(ratio : Tensor, advantage : Tensor) -> Tensor:
    unclipped = torch.mean(ratio*advantage)
    clipped = torch.mean(torch.clip(ratio, 1-epsilon, 1+epsilon)*advantage)
    return -torch.min(unclipped, clipped)    # - : we try to minimize loss when backprop

def value_loss_fn(advantage : Tensor) -> Tensor:
    return torch.mean(advantage**2)





### training
num_episodes = 1000
num_timesteps = 200

print("Chapter 13.    PPO")

for i in range(num_episodes):
    state = env.reset()
    ep_means = []    # this can be removed, in that case this must be recomuted by policy network at training step
    ep_log_stds = []    # this can be removed, in that case this must be recomuted by policy network at training step
    ep_states = []
    ep_actions = []
    ep_rewards = []

    for t in range(num_timesteps):
        mean, log_std = policy_network(tensor(state).to(devie=device))
        std = torch.exp(std)

        action = torch.normal(mean, std).to(device=cpu).detach().clamp(action_bound[0], action_bound[1]).numpy()
        next_state, reward, done, truncated, info = env.step(action)

        ep_means.append(mean)
        ep_log_stds.append(log_std)
        ep_states.append(state)
        ep_actions.append(action)
        ep_rewards.append(reward)

        state = next_state

        if (t+1%batch_size == 0) or (t == num_timesteps-1):
            value = value_network(tensor(next_state, dtype=torch.float, device=device))

            # compute ratio and advanatage
            discounted_rewards = []
            for r in ep_rewards[::-1]:
                value = r + gamma*value
                discounted_rewards.append(value)
            discounted_rewards.reverse()
            
            ep_means = torch.tensor(ep_means, dtype=torch.float, device=device)
            ep_log_stds = torch.tensor(ep_log_stds, dtype=torch.float, device=device)
            ep_actions = torch.tensor(ep_actions, dtype=torch.float, device=device)

            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float, device=device)
            values = value_network(tensor(ep_states, dtype=torch.float, device=device))

            ratios = torch.exp(policy_network.log_prob(ep_actions, ep_means, ep_log_stds) - \
                               old_policy_network.log_prob(ep_actions, ep_means, ep_log_stds))
            advanatage = discounted_rewards-values
            
            # train
            policy_loss = objective_fn(ratios, advanatage)
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            value_loss = value_loss_fn(advanatage)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
            
            # reset buffer
            ep_means = [] 
            ep_log_stds = []
            ep_states = []
            ep_actions = []
            ep_rewards = []



            
            


