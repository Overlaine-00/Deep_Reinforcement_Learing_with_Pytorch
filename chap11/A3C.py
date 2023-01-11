import os
import warnings
warnings.filterwarnings('ignore')

import gym
import numpy as np

import torch
from torch import tensor
from torch import nn, optim, Tensor
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.multiprocessing as mp
# mp.set_start_method('spawn')

cpu = "cpu"
device = cpu # "cuda" if torch.cuda.is_available() else "cpu"

import matplotlib.pyplot as plt




### create environment
env = gym.make('MountainCarContinuous-v0')

state_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]
action_bound = [env.action_space.low, env.action_space.high]




# variables
num_episodes = 2000
num_timesteps = 200
update_global = 10

global_episode = 0

gamma = 0.98
beta = 0.01
log_dir = 'D:\\logs'




### actor-critic model A3C

class Actor_pytorch(nn.Module):
    def __init__(self, state_shape=state_shape, action_shape=action_shape, device=device):
        super().__init__()
        self.state_shape, self.action_shape = state_shape, action_shape
        self.device = device
        
        self.initial = nn.Linear(self.state_shape, 200, device=device, dtype=torch.float)
        self.mean = nn.Linear(200, self.action_shape, device=device, dtype=torch.float)
        self.sigma = nn.Linear(200, self.action_shape, device=device, dtype=torch.float)
    
    def forward(self, state):
        # can be replaced by Categorical
        state = tensor(state, dtype=torch.float, device=self.device)
        stat = F.relu(self.initial(state))
        mean = F.tanh(self.mean(stat))
        std = F.softplus(self.sigma(stat)) + 1e-8
        
        return mean, std


class Critic_pytorch(nn.Module):
    def __init__(self, state_shape=state_shape, action_shape=action_shape, device=device):
        super().__init__()
        self.state_shape, self.action_shape = state_shape, action_shape
        self.device = device
        
        self.value_1 = nn.Linear(self.state_shape, 100, dtype=torch.float, device=self.device)
        self.vaule_2 = nn.Linear(100,1, dtype=torch.float, device=self.device)
    
    def forward(self, state) -> Tensor:
        """
        Output: value (V(s_t))
        """
        state = tensor(state, dtype=torch.float, device=self.device)
        value = F.relu(self.value_1(state))
        value = self.vaule_2(value)
        
        return value




class A3C_pytorch:
    def __init__(self, global_actor : Actor_pytorch, global_critic : Critic_pytorch, state_shape=state_shape, action_shape=action_shape, device=device):
        self.global_actor, self.global_critic = global_actor, global_critic
        self.state_shape, self.action_shape = state_shape, action_shape
        self.device = device
        self.buffer_size = 100000
        
        self.actor_network = Actor_pytorch(state_shape=state_shape, action_shape=action_shape, device=device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=1e-3)
        
        self.critic_network = Critic_pytorch(state_shape=state_shape, action_shape=action_shape, device=device)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=1e-4)
        
        self.reset_transition()
    
    
    def reset_transition(self):
        self.batch_log_prob = None
        self.batch_entropy = None
        self.batch_value = None
        self.batch_next_value = None
        self.batch_reward = []
    
    
    def store_transition(self, log_prob : Tensor, entropy : Tensor, value : Tensor, reward, next_value : Tensor):
        if self.batch_log_prob == None: self.batch_log_prob = log_prob.unsqueeze(dim=0)
        else: self.batch_log_prob = torch.cat([self.batch_log_prob, log_prob.unsqueeze(dim=0)])
        
        if self.batch_entropy == None: self.batch_entropy = entropy.unsqueeze(dim=0)
        else: self.batch_entropy = torch.cat([self.batch_entropy, entropy.unsqueeze(dim=0)])
        
        if self.batch_value == None: self.batch_value = value.unsqueeze(dim=0)
        else: self.batch_value = torch.cat([self.batch_value, value.unsqueeze(dim=0)])
        
        self.batch_reward.append(reward)
        
        if self.batch_next_value == None: self.batch_next_value = next_value.unsqueeze(dim=0)
        else: self.batch_next_value = torch.cat([self.batch_next_value, next_value.unsqueeze(dim=0)])
    
    
    def select_action(self, state):
        '''
            /gym/envs/classic_control/continuous_mountain_car.py
            -> For input variable action, gym uses action[0]
               action.item()[0] raises error, so return action
               instead of action.item() .
        '''
        mean, std = self.actor_network(state)
        value = self.critic_network(state)
        
        sample_space = torch.distributions.Normal(mean, std) # can be replaced by Categorical
        action = sample_space.sample()
        log_prob = -sample_space.log_prob(action)
        entropy = sample_space.entropy()
        return action, log_prob, value, entropy
    
    
    def loss_fn(self, log_prob : Tensor, entropy : Tensor, value : Tensor, target_Q : Tensor):
        """
        Inputs:
                log_prob : from A3C_pytorch.select_action
                value : V(t)
                target_Q : Q(s_t,a_t) = r_t + V(t+1)
        """
        td_error = torch.subtract(target_Q, value)
        critic_loss = F.mse_loss(td_error, torch.zeros_like(td_error))
        actor_loss = log_prob*td_error.detach() + beta*entropy.detach()
        
        return actor_loss, critic_loss
    
    
    def train(self):
        self.actor_network.train(); self.critic_network.train()
        self.global_actor.train(); self.global_critic.train()
        
        rewards = tensor(self.batch_reward, dtype=torch.float, device=self.device)
        for i in range(len(self.batch_reward)):
            actor_loss, critic_loss = self.loss_fn(self.batch_log_prob[i],
                                                   self.batch_entropy[i],
                                                   self.batch_value[i],
                                                   rewards[i]+self.batch_next_value[i])
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            for global_param, local_param in zip(self.global_critic.parameters(), self.critic_network.parameters()):
                global_param._grad = local_param.grad
            self.critic_optimizer.step()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            for global_param, local_param in zip(self.global_actor.parameters(), self.actor_network.parameters()):
                global_param._grad = local_param.grad
            self.actor_optimizer.step()
            
            self.actor_network.load_state_dict(self.global_actor.state_dict())
            self.critic_network.load_state_dict(self.global_critic.state_dict())

        self.reset_transition()
    



# train and test (global actor only)
def train(local_network : A3C_pytorch, reward_queue : mp.Queue):
    global global_episode
    env = gym.make("MountainCarContinuous-v0")
    total_step = 1
    
    while global_episode < num_episodes:
        state = env.reset()
        local_network.reset_transition()
        Log_prob = Entropy = Value = Reward = None
        Return = 0
        episode_reward = 0
        
        for t in range(num_timesteps):
            action, log_prob, value, entropy = local_network.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            if t>0: local_network.store_transition(Log_prob, Entropy, Value, Reward, value)
            Log_prob, Entropy, Value, Reward = log_prob, entropy, value, reward
            if done: local_network.store_transition(Log_prob, Entropy, Value, Reward, torch.zeros_like(Value))
            
            if total_step > 1 and total_step % update_global == 1:
                local_network.train()

            total_step += 1
            episode_reward += reward
            if done:
                reward_queue.put(episode_reward)
                Return += episode_reward
                break
            state = next_state
        
        if (global_episode+1)%10 == 0:
            print(f"Train episode {global_episode-8}~{global_episode+1}, reward {Return}")
            Return = 0
        global_episode += 1


def test(global_actor : Actor_pytorch, reward_queue : mp.Queue):
    global_actor.eval()
    with torch.no_grad():
        env = gym.make("MountainCarContinuous-v0")
        
        Return = 0
        for i in range(num_episodes//10):
            state = env.reset()
            
            done = False
            step = 0
            episode_reward = 0
            while not done and step < num_timesteps:
                mean, std = global_actor(Tensor(state, device=device))
                action = torch.distributions.Normal(mean, std).sample()
                
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                step += 1
                episode_reward += reward

            Return += episode_reward
            reward_queue.put(episode_reward)
            if i>0 and (i+1)%10 == 0:
                print(f"Test episode {i-8}~{i+1}, reward {Return}")
                Return = 0





# not used
class worker_pytorch(mp.Process):
    def __init__(self, global_actor : Actor_pytorch, global_critic : Critic_pytorch, index : int):
        super().__init__()
        self.name = f"Worker {index}"
        
        self.env = gym.make("MountainCarContinuous-v0").unwrapped
        
        self.network = A3C_pytorch(global_actor, global_critic)
    
    
    def run(self):
        global global_episode
        total_step = 1
        while global_episode < num_episodes:
            state = self.env.reset()[0]
            Return = 0
            self.network.reset_transition()
            Log_prob = Entropy = Value = Reward = None
            
            for t in range(num_timesteps):
                action, log_prob, value, entropy = self.network.select_action(state)
                next_state, reward, done, truncated, info = self.env.step(action)
                
                if t>0: self.network.store_transition(Log_prob, Entropy, Value, Reward, value)
                Log_prob, Entropy, Value, Reward = log_prob, entropy, value, reward
                if done: self.network.store_transition(Log_prob, Entropy, Value, Reward, torch.zeros_like(Value))
                
                if total_step > 1 and total_step % update_global == 1:
                    self.network.train()

                total_step += 1
                Return += reward
                if done:
                    # print("")
                    break
                state = next_state
                
            global_episode += 1
            




### training
global_actor, global_critic = Actor_pytorch(), Critic_pytorch()
global_actor.share_memory(), global_critic.share_memory()
global_reward_queue = mp.Queue()

print(f"\nChapter 11 : Actor-Critic model (A3C).    Device : {device}\n")

num_process = mp.cpu_count()
processes : list[mp.Process] = []
local_workers = [A3C_pytorch(global_actor, global_critic) for _ in range(num_process-1)]
local_reward_queue = [mp.Queue() for _ in range(num_process-1)]
for rank in range(num_process):
    if rank == 0:
        p = mp.Process(target=test, args=(global_actor, global_reward_queue,))
    else:
        p = mp.Process(target=train, args=(local_workers[rank-1], local_reward_queue[rank-1],))
    p.start()
    processes.append(p)
for p in processes:
    p.join()
