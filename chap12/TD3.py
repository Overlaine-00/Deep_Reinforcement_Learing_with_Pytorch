import numpy as np
import gym

import torch
from torch import tensor
from torch import nn, optim, Tensor
import torch.nn.functional as F

cpu = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"



### create environment
env = gym.make("Pendulum-v1", g=9.81).unwrapped

state_shape = env.observation_space.shape[0]    # 3
action_shape = env.action_space.shape[0]    # 1
action_bound = np.concatenate([env.action_space.low, env.action_space.high], dtype=np.float32)    # [-2,2]


# variables
gamma = 0.9    # decay
tau = 0.005    # lr
buffer_size = int(10e4)
batch_size = 64
action_update_delay = 5





### TD3 modules    not trained -> see https://github.com/sfujim/TD3/blob/master/TD3.py

## networks
# Actor
class Actor(nn.Module):
    def __init__(self, state_shape = state_shape, action_shape = action_shape, upper_bound = action_bound[1]):
        super().__init__()
        self.upper_bound = upper_bound
        
        self.layer1 = nn.Linear(state_shape, 400, dtype=torch.float)
        self.layer2 = nn.Linear(400, 300, dtype=torch.float)
        self.layer3 = nn.Linear(300, action_shape, dtype=torch.float)
    
    def forward(self, state : Tensor) -> Tensor:
        val = torch.relu(self.layer1(state))
        val = torch.relu(self.layer2(val))
        val = torch.tanh(self.layer3(val))
        # add random selection
        val = torch.clamp(val + torch.normal(0.0,tensor(3.0)), action_bound[0], action_bound[1])
        return self.upper_bound*val

# Critic
class Critic(nn.Module):
    def __init__(self, state_shape=state_shape, action_shape=action_shape):
        super().__init__()
        
        # more channels than textbook, add normalize
        self.layer1 = nn.Linear(state_shape, 400, dtype=torch.float)
        self.normalize = nn.LayerNorm(400, dtype=torch.float)
        self.layer2 = nn.Linear(400+action_shape, 300, dtype=torch.float)
        self.layer3 = nn.Linear(300, 1, dtype=torch.float)
    
    def forward(self, state : Tensor, action : Tensor) -> Tensor:
        """
        Input : s_t, a_t
        Output : Q(s_t, a_t)
        """
        val = F.relu(self.layer1(state))
        val = self.normalize(val)
        val = F.relu(self.layer2(torch.cat([val, action], dim=1)))    # input.shape = (batch_size, state/action_shape)
        val = self.layer3(val)
        return val




## storage
# replay buffer
state_index = [_ for _ in range(state_shape)]
action_index = [_ + state_shape for _ in range(action_shape)]
reward_index = [state_shape + action_shape]
next_state_index = [_ + state_shape + action_shape + 1 for _ in range(state_shape)]
class replay_buffer:
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




## used functions
# parameter update
def soft_update(source : Actor or Critic, target : Actor or Critic, lr : float = tau):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(target_param.data*(1-lr) + source_param.data*lr)

def hard_update(source : Actor or Critic, target : Actor or Critic):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(source_param.data)


# loss
"""
Input data size : (batch_size, each shape)
    for example, action.size() = (batch_size, action_shape)
    Make sure that (batch_size, ) must be converted to (batch_size, 1) when treating reward
    I use reward = sample_data[[idx]].
"""
def actor_loss_function(state : Tensor, main_actor : Actor, target_critic : Critic) -> Tensor:
    # state tensor storged at device
    action = main_actor(state)
    temp = target_critic(state, action)
    return -temp.mean()

def critic_loss_function(state : Tensor, action : Tensor,
                reward : Tensor, next_state : Tensor,
                target_actor : Actor,
                main_critic : Critic, target_critic : Critic,
                device = device):
    expected_Q = main_critic(state, action)
    with torch.no_grad():
        actual_Q = reward.to(device=device) + \
                   gamma*target_critic(next_state, target_actor(next_state))

    return F.mse_loss(expected_Q, actual_Q, reduction='mean')




## used variable
# network and buffer
main_actor, target_actor = Actor().to(device=device), Actor().to(device=device)
main_critic = [Critic().to(device), Critic().to(device=device)]
target_critic = [Critic().to(device), Critic().to(device=device)]

hard_update(main_actor, target_actor)
for i in range(2) : hard_update(main_critic[i], target_critic[i])

buffer = replay_buffer()

# optimizer
actor_optimizer = optim.Adam(main_actor.parameters(), lr=0.001)
critic_optimizer = [optim.Adam(critic.parameters(), lr=0.005) for critic in main_critic]


## training
def train(train_actor : bool = True):
    sample_data = tensor(buffer.batch_sample(batch_size), dtype=torch.float).to(device=device)

    critic_losses = [critic_loss_function(sample_data[:,state_index],
                                          sample_data[:,action_index],
                                          sample_data[:,reward_index],
                                          sample_data[:,next_state_index],
                                          target_actor,
                                          _main_critic, _target_critic)
                     for _main_critic, _target_critic in zip(main_critic, target_critic)]
    
    # update critic network having maximum loss
    maximal_critic_index = np.argmax(np.array([loss.item() for loss in critic_losses]))    # max? sum?
    critic_loss = critic_losses[maximal_critic_index]
    
    #prev_weight = main_critic[maximal_critic_index].layer3.weight.data.clone()
    
    main_critic[maximal_critic_index].zero_grad()
    critic_loss.backward()
    critic_optimizer[maximal_critic_index].step()
    
    #print(torch.max(torch.abs(prev_weight - main_critic[maximal_critic_index].layer3.weight.data)))
    
    soft_update(main_critic[maximal_critic_index], target_critic[maximal_critic_index])
    
    
    if train_actor:
        # may use any target critic network.
        actor_loss = actor_loss_function(sample_data[:,state_index], main_actor, target_critic[maximal_critic_index])
        
        main_actor.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        soft_update(main_actor, target_actor)
    




### training step
num_epsiodes = 500
num_timesteps = 500

print("Chapter 12.    TD3")

ep10_rewards = 0
for i in range(num_epsiodes):
    state = env.reset()
    
    for t in range(num_timesteps):
        action = main_actor(tensor(state).to(device=device)).to(device=cpu).item()
        next_state, reward, done, info = env.step(action)
        
        if done: break
        buffer.append(state, action, reward, next_state)
        state = next_state
        
        ep10_rewards += reward
    
    if (i+1)%5 == 0: train(True)
    else: train(False)
    
    if (i+1)%10==0:
        print(f"Episode {i-9}~{i+1}, average return {ep10_rewards/10}")
        ep10_rewards = 0
