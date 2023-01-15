import numpy as np
import gym

import torch
from torch import tensor
from torch import nn, optim, Tensor, tensor, autograd
import torch.nn.functional as F

cpu = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("Chapter 13.    PPO")



### create environment
env = gym.make("Pendulum-v1", g=9.81).unwrapped

state_shape = env.observation_space.shape[0]    # 3
action_shape = env.action_space.shape[0]    # 1
action_bound = np.concatenate([env.action_space.low, env.action_space.high], dtype=np.float32)    # [-2,2]


# variables
epsilon = 0.2
gamma = 0.9    # decay
beta_initial = 0.1    # penalty coefficient in PPO-penalty
lr = 0.001
buffer_size = int(10e4)
batch_size = 32

mode = input("Select PPO mode -- clipped / penalty : ")
print("Selected mode :", mode)





### PPO
class Policy(nn.Module):
    def __init__(self, state_shape = state_shape, action_shape = action_shape, mode: str = mode):
        '''
        mode : 'clipped' or 'penalty'
        '''
        super().__init__()
        self.mode = mode

        self.mutual_network = nn.Sequential(nn.Linear(state_shape, 100), nn.ReLU())
        self.mean = nn.Sequential(nn.Linear(100, action_shape), nn.Tanh())    # 2* : action bound = [-2,2] whereas tanh(x) \in (-1,1)
        self.log_std = nn.Linear(100, action_shape)

        if mode == "penalty":
            self.beta = beta_initial
    

    def get_beta(self): return self.beta

    def log_prob(self, action : Tensor, mean : Tensor, log_std : Tensor) -> Tensor:
        return -log_std - (action-mean)**2 / (2*torch.exp(log_std)**2)
    

    def forward(self, state : Tensor) -> tuple[Tensor]:
        out = self.mutual_network(state)
        mean = 2*self.mean(out)
        log_std = self.log_std(out)

        return mean, log_std


class Value(nn.Module):
    def __init__(self, state_shape = state_shape):
        super().__init__()

        self.value = nn.Sequential(nn.Linear(state_shape, 100), nn.ReLU(),
                                   nn.Linear(100, 1))
    

    def forward(self, state : Tensor) -> Tensor:
        return self.value(state)






## used variables
policy_network = Policy(); policy_network.to(device=device, dtype=torch.float)
old_policy_network = Policy(); old_policy_network.to(device=device, dtype=torch.float)
value_network = Value(); value_network.to(device=device, dtype=torch.float)

policy_optimizer = optim.Adam(policy_network.parameters(), lr=lr)
value_optimizer = optim.Adam(value_network.parameters(), lr=lr)


def update_old():
    for old_param, param in zip(old_policy_network.parameters(), policy_network.parameters()):
        old_param.data.copy_(param.data)

def KL_div(old_mean: Tensor, old_log_std: Tensor, mean: Tensor, log_std: Tensor):
    old_std, std = torch.exp(old_log_std), torch.exp(log_std)
    return 1/2*torch.sum( old_std/std + (mean-old_mean)**2/std + log_std/old_log_std, dim=1)


def objective_fn(ratio : Tensor, advantage : Tensor, kl_div : Tensor = None,
                 beta = policy_network.get_beta(), mode=mode) -> Tensor:
    if mode == "clipped":
        unclipped = torch.mean(ratio*advantage)
        clipped = torch.mean(torch.clamp(ratio, 1-epsilon, 1+epsilon)*advantage)
        return -torch.min(unclipped, clipped)    # - : we try to minimize loss when backprop
    elif mode == "penalty":
        return torch.mean(ratio*advantage - beta*kl_div)

def value_loss_fn(advantage : Tensor) -> Tensor:
    return torch.mean(advantage**2)

update_old()





### training
num_episodes = 1000
num_timesteps = 200

Return = 0
for i in range(num_episodes):
    state = env.reset()[0]
    ep_states = []
    ep_actions = []
    ep_rewards = []

    for t in range(num_timesteps):
        mean, log_std = policy_network(tensor(state, device=device, dtype=torch.float))
        std = torch.exp(log_std)

        action = torch.normal(mean, std).to(device=cpu).detach().clamp(action_bound[0], action_bound[1]).numpy()
        next_state, reward, done, truncated, info = env.step(action)

        ep_states.append(state)
        ep_actions.append(action)
        ep_rewards.append(reward/8+1)

        Return += reward
        state = next_state

        if (t+1%batch_size == 0) or (t == num_timesteps-1):
            value = value_network(tensor(next_state, device=device, dtype=torch.float))

            # compute ratio and advantage
            discounted_rewards = []
            for r in ep_rewards[::-1]:
                value = r + gamma*value
                discounted_rewards.append(value)
            discounted_rewards.reverse()

            ep_states, ep_actions, ep_rewards = np.array(ep_states), np.array(ep_actions), np.array(ep_rewards)
            
            ep_states = tensor(ep_states, device=device, dtype=torch.float)
            ep_means, ep_log_stds = policy_network(ep_states)
            ep_old_means, ep_old_log_stds = old_policy_network(ep_states)
            ep_actions = tensor(ep_actions, device=device, dtype=torch.float)

            discounted_rewards = tensor(discounted_rewards, device=device, dtype=torch.float)
            values : Tensor = value_network(ep_states)
            
            ratios = torch.exp(policy_network.log_prob(ep_actions, ep_means, ep_log_stds) - \
                               old_policy_network.log_prob(ep_actions, ep_old_means, ep_old_log_stds))
            advantage = discounted_rewards-values
            if (mode == "penalty"):
                kl_div = KL_div(ep_old_means, ep_old_log_stds, ep_means, ep_log_stds)
            else:
                kl_div = None

            # train
            value_loss = value_loss_fn(advantage)
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            policy_loss = objective_fn(ratios, advantage.detach(), kl_div)
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            update_old()
            
            # reset buffer
            ep_states = []
            ep_actions = []
            ep_rewards = []

    if ((i+1)%10 == 0):
        print(f"Episode {i+1}, average return {Return}")
    
    Return = 0


            
            


