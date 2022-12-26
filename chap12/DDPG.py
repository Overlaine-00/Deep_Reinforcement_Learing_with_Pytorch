import warnings
warnings.filterwarnings('ignore')

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
gamma = 0.9
tau = 0.005
replay_buffer = int(10e5)
batch_size = 64



### DDPG class. see : https://github.com/ghliu/pytorch-ddpg
class Actor_pytorch(nn.Module):
    def __init__(self, state_shape=state_shape, action_shape=action_shape, upper_bound = action_bound[1], device=device):
        super().__init__()
        self.upper_bound = upper_bound
        self.device = device

        # one more layer, more channels than textbook
        self.layer1 = nn.Linear(state_shape, 400, device=self.device, dtype=torch.float)
        self.layer2 = nn.Linear(400, 300, device=self.device, dtype=torch.float)
        self.layer3 = nn.Linear(300, action_shape, device=self.device, dtype=torch.float)
    
    def forward(self, state):
        """
        Output : a_t
                 lower_action_bound(=-2) <= a_t <= upper_action_bound(=+2)
                 To restrtict range(-1~1), activation tanh (instead of ReLU) is used
        """
        state = tensor(state, dtype=torch.float, device=device)
        val = F.relu(self.layer1(state))
        val = F.relu(self.layer2(val))
        val = F.tanh(self.layer3(val))
        return self.upper_bound*val


class Critic_pytorch(nn.Module):
    def __init__(self, state_shape=state_shape, action_shape=action_shape, device=device):
        super().__init__()
        self.state_shape, self.action_shape = state_shape, action_shape
        self.device=device
        
        # more channels than textbook, add normalize
        self.layer1 = nn.Linear(state_shape, 400, device=self.device, dtype=torch.float)
        self.normalize = nn.LayerNorm(400, device=self.device, dtype=torch.float)
        self.layer2 = nn.Linear(400+action_shape, 300, device=self.device, dtype=torch.float)
        self.layer3 = nn.Linear(300, 1, device=self.device, dtype=torch.float)
    
    def forward(self, state, action):
        """
        Input : s_t, a_t
        Output : Q(s_t, a_t)
        """
        action = tensor(action, device=self.device, dtype=torch.float)
        state = tensor(state, device=self.device, dtype=torch.float)
        
        val = F.relu(self.layer1(state))
        val = self.normalize(val)
        val = F.relu(self.layer2(torch.cat([val, action], dim=1)))
        val = self.layer3(val)
        return val





class DDPG_pytorch:
    def __init__(self, state_shape = state_shape, action_shape = action_shape, action_bound = action_bound,
                 buffer_size = replay_buffer, batch_size = batch_size, device=device):
        self.state_shape, self.action_shape = state_shape, action_shape
        self.action_low, self.action_high = action_bound    # tensor(action_bound, device=device)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        
        self.reset_transition()
        
        self.main_actor = Actor_pytorch(state_shape, action_shape, self.action_high, device)
        self.target_actor = Actor_pytorch(state_shape, action_shape, self.action_high, device)
        self.main_critic = Critic_pytorch(state_shape, action_shape, device)
        self.target_critic = Critic_pytorch(state_shape, action_shape, device)
        self.actor_optimizer = optim.Adam(self.main_actor.parameters(), 0.001)
        self.critic_optimizer = optim.Adam(self.main_critic.parameters(), 0.005)
        for source_param, target_param in zip(self.main_critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(source_param.data)
        for source_param, target_param in zip(self.main_actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(source_param.data)
    
    
    def reset_transition(self):
        self.state_buffer = np.zeros([self.buffer_size, self.state_shape], dtype=np.float32)
        self.action_buffer = np.zeros([self.buffer_size, self.action_shape], dtype=np.float32)
        self.reward_buffer = np.zeros([self.buffer_size, 1], dtype=np.float32)
        self.next_state_buffer = np.zeros([self.buffer_size, self.state_shape], dtype=np.float32)
        self.record_num = 0
    
    def store_transition(self, state : np.ndarray, action : np.ndarray or np.float32, reward : np.float32, next_state : np.ndarray):
        index = self.record_num % self.buffer_size
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state
        self.record_num += 1
    
    
    def update_target_network(self, learning_rate = tau):
        for source_param, target_param in zip(self.main_critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(target_param.data*(1-learning_rate) + source_param.data*learning_rate)
        for source_param, target_param in zip(self.main_actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(target_param.data*(1-learning_rate) + source_param.data*learning_rate)
    
    
    def select_action(self, state) -> np.float32:
        action = self.main_actor(state).item()
        action += np.random.normal(0,3)
        action = np.clip(action, self.action_low, self.action_high)
        
        return action
    
    
    def actor_loss(self, state : Tensor) -> Tensor:
        action = self.main_actor(state)
        loss = -self.main_critic(state, action).mean()
        return loss

    def critic_loss(self, state : Tensor, action : Tensor, reward : Tensor, next_state : Tensor) -> Tensor:
        main_Q = self.main_critic(state, action)
        next_action = self.target_actor(next_state)
        target_Q = reward + gamma*self.target_critic(next_state, next_action)
        loss = F.mse_loss(main_Q, target_Q)
        return loss
    
    
    def train(self):
        # self.main_actor.train(); self.main_critic.train()
        # self.target_actor.eval(); self.target_critic.eval()
        
        random_index = np.random.choice(min(self.record_num, self.buffer_size), self.batch_size)
        state_buffer = torch.from_numpy(self.state_buffer[random_index]).to(device=device)
        action_buffer = torch.from_numpy(self.action_buffer[random_index]).to(device=device)
        reward_buffer = torch.from_numpy(self.reward_buffer[random_index]).to(device=device)
        next_state_buffer = torch.from_numpy(self.next_state_buffer[random_index]).to(device=device)
        
        critic_loss = self.critic_loss(state_buffer, action_buffer, reward_buffer, next_state_buffer)
        self.main_critic.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = self.actor_loss(state_buffer)
        self.main_actor.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        
        # target network parameter update. maybe less frequent than present
        # self.target_actor.train(); self.target_critic.train()
        self.update_target_network()



### training
ddpg = DDPG_pytorch()
num_episodes = 500
num_timesteps = 500
# update_period = 10

print(f"\nChapter 12 : DDPG.    Device : {device}\n")

rewards_sum = 0.0
for i in range(num_episodes):
    state = env.reset()
    
    for t in range(num_timesteps):
        '''
        correct u = np.clip(u, -self.max_torque, self.max_torque)[0]
             -> u = np.clip(u, -self.max_torque, self.max_torque)
        in anaconda3/envs/RL/lib/python3.9/site-packages/gym/envs/classic_control/pendulum.py
           (line 40, def step(): ~~)
        '''
        action = ddpg.select_action(state)
        next_state, reward, done, info = env.step(np.array([action],dtype=np.float32))

        ddpg.store_transition(state, action, reward, next_state)
        rewards_sum += reward
        
        state = next_state
        if done: break
    
    ddpg.train()
    if (i+1)%10 == 0:
        print(f"Episode {i-8}~{i+1},    Average return {rewards_sum/10}")
        rewards_sum = 0.0
env.close()
