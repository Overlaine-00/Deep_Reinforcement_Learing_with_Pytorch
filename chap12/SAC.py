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
tau = 0.01    # lr
alpha = 0.1    # weight of entropy
buffer_size = int(10e4)
batch_size = 100





### SAC modules

## networks
# Actor
class Actor(nn.Module):
    
    def __init__(self, state_shape = state_shape, action_shape = action_shape):
        super().__init__()
        
        # mean
        self.mean1 = nn.Linear(state_shape, 100, dtype=torch.float)
        self.mean2 = nn.Linear(100, action_shape, dtype=torch.float)
        
        # std should be larger than 0 -> get log(std) and take exponential
        self.log_std1 = nn.Linear(state_shape, 200, dtype=torch.float)
        self.log_std2 = nn.Linear(200, action_shape, dtype=torch.float)
        
    def forward(self, state : Tensor) -> Tensor:
        m = torch.relu(self.mean1(state))
        m = torch.tanh(self.mean2(m))
        
        std = torch.relu(self.log_std1(state))
        std = self.log_std2(std)
        std = torch.exp(std)
        
        return m, std
        
# Critic
class Critic_V(nn.Module):
    def __init__(self, state_shape = state_shape, action_shape = action_shape):
        super().__init__()
        
        self.layer1 = nn.Linear(state_shape, 300, dtype=torch.float)
        self.layer2 = nn.Linear(300, 200, dtype=torch.float)
        self.layer3 = nn.Linear(200,action_shape, dtype=torch.float)
        
    def forward(self, state : Tensor) -> Tensor:
        val = torch.relu(self.layer1(state))
        val = torch.relu(self.layer2(val))
        val = self.layer3(val)
        
        return val

class Critic_Q(nn.Module):
    def __init__(self, state_shape = state_shape, action_shape = action_shape):
        super().__init__()
        
        self.layer1 = nn.Linear(state_shape, 200, dtype=torch.float)
        self.normalize = nn.BatchNorm1d(200, dtype=torch.float)
        self.layer2 = nn.Linear(200+state_shape, 200, dtye=torch.float)
        self.layer3 = nn.Linear(200, action_shape, dtype=torch.float)
        
    def forward(self, state : Tensor, action : Tensor) -> Tensor:
        s = torch.relu(self.layer1(state))
        s = self.normalize(s)
        val = torch.relu(self.layer2(torch.cat([s, action], dim=1)))    # input.shape = (batch_size, state/action_shape)
        val = self.layer3(val)
        
        return val





## storage
# replay buffer
state_index = [_ for _ in range(state_shape)]
action_index = [_ + state_shape for _ in range(action_shape)]
log_entropy_index = [state_shape + 1]
reward_index = [state_shape + action_shape + 1]
next_state_index = [_ + state_shape + action_shape + 2 for _ in range(state_shape)]
class replay_buffer:
    def __init__(self, capacity : int = buffer_size, state_shape : int = state_shape, action_shape : int = action_shape):
        '''
        basic replay buffer.
        store : state, action, reward, next_state
        '''
        self.capacity = capacity
        self.pointer = 0
        
        self.buffer = np.zeros([capacity, 2*state_shape + action_shape + 2], dtype=np.float32)
    
    def append(self, state : np.ndarray, action : np.float32, log_entropy : np.float32, reward : np.float32, next_state : np.ndarray):
        index = self.pointer%self.capacity
        self.buffer[index] = np.concatenate([state, np.array([action, log_entropy, reward]), next_state])
        
        self.pointer += 1
        if self.pointer >= 2*self.capacity: self.pointer -= self.capacity
        
        del index
    
    def batch_sample(self, batch_size : int) -> np.ndarray:
        # choose sample by bootstapping
        index = np.random.randint(0, min(self.pointer, self.capacity), (batch_size,))
        
        return self.buffer[index]
    



## used functions
# parameter update
def soft_update(source, target, lr : float = tau):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(target_param.data*(1-lr) + source_param.data*lr)

def hard_update(source, target):
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(source_param.data)


# loss
def actor_loss(state : Tensor, log_entropies : Tensor, actor : Actor, critic_Q : Critic_Q):
    '''
    expected : Q(s,actor(s)) - alpha*,   shape = (batch_size,1)
    loss : -mean
    '''
    expected = critic_Q(state, actor(state)) - alpha*log_entropies
    
    return -torch.mean(expected)

def critic_v_loss(state : Tensor, action : Tensor, log_entropies : Tensor, critic_V : Critic_V, critic_Q : Critic_Q or list[Critic_Q]):
    '''
    expected : V(s),   shape = (batch_size,1)
    actual(target) : min_i(Q_i(s,a)) - alpha*entropy(a|s),    shape = (batch_size,1)
    loss : MSE
    '''
    if type(critic_Q) == list:    # choose minimal Q-value along input networks
        expected = min([critic(state, action) for critic in critic_Q] - alpha*log_entropies)
    else:
        expected = critic_Q(state, action) - alpha*log_entropies
    
    return F.mse_loss(expected - critic_V(state))

def critic_q_loss(state : Tensor, action : Tensor, reward : Tensor, next_state : Tensor,
                  critic_V : Critic_V, critic_Q : Critic_Q):
    '''
    expected : Q(s,a),   shape = (batch_size,1)
    actual(target) : r + V'(s'),   shape = (batch_size,1)
    (V' : target network, s' : next state)
    loss : MSE
    '''
    expected = critic_Q(state, action)
    actual = reward + gamma*critic_V(next_state)
    
    return F.mse_loss(expected - actual)



## used variables
# network and buffer
Q_num = 2

main_actor = Actor().to(device=device)
main_value, target_value = Critic_V().to(device=device), Critic_V().to(device=device)
critic_q = [Critic_Q().to(device=device) for _ in range(Q_num)]

hard_update(main_value, target_value)

buffer = replay_buffer()

# optimizer
actor_optimizer = optim.Adam(main_actor.parameters(), lr=0.001)
value_optimizer = optim.Adam(main_value.parameters, lr=0.005)
critic_q_optimizer = [optim.Adam(critic.parameters(), lr=0.005) for critic in critic_q]


## training
def train():
    sample_data = tensor(buffer.batch_sample(batch_size), dtype=torch.float).to(device=device)
    
    # value network update
    value_loss = critic_v_loss(sample_data[:,state_index],
                               sample_data[:,action_index],
                               sample_data[:,log_entropy_index],
                               main_value, critic_q)
    
    value_optimizer.zero_grad()    # main_value.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    
    # Q network update
    q_value_loss = [critic_q_loss(sample_data[:,state_index],
                                 sample_data[:,action_index],
                                 sample_data[:,reward_index],
                                 sample_data[:,next_state_index],
                                 target_value, _critic_q)
                    for _critic_q in critic_q]
    
    for i in range(Q_num):
        critic_q_optimizer[i].zero_grad()    # critic_q[i].zero_grad()
        q_value_loss[i].backward()
        critic_q_optimizer[i].step()
        
    # actor network update
    loss = actor_loss(sample_data[:,state_index],
                      sample_data[:,log_entropy_index],
                      main_actor, critic_q[0])    # can use any of critic_q
    
    actor_optimizer.zero_grad()    # main_actor.zero_grad()
    loss.backward()
    actor_optimizer.step()
    
    # update target network
    soft_update(main_value, target_value)






### training step
num_epsiodes = 500
num_timesteps = 500

print("Chapter 12.    SAC")

ep10_rewards = 0
for i in range(num_epsiodes):
    state = env.reset()
    
    for t in range(num_timesteps):
        mean, std = main_actor(tensor(state).to(device=device)).to(device=cpu).item()

        action = torch.normal(mean, std).clamp(action_bound[0], action_bound[1])
        log_entropy = -torch.prod(std)**2 - torch.sum(((action-mean)/std)**2)

        action = action.to(device=cpu).clone().numpy()
        log_entropy = log_entropy.to(device=cpu).clone().numpy()
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