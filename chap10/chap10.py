import numpy as np

import gym

import torch
from torch import tensor
from torch import nn, optim, Tensor
from torch.nn import Sequential
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

import plotly.graph_objects as go
import plotly.io as pio

cpu = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()




# define environment
env = gym.make('CartPole-v0')
state_shape = env.observation_space.shape[0]
num_actions = env.action_space.n

# normalization
gamma = 0.95
def discount_and_normalize_rewards(episode_rewards : list[float]) -> Tensor:
    discounted_rewards = np.zeros_like(episode_rewards, dtype=np.float64)
    reward_to_go = 0.0
    for i in reversed(range(len(episode_rewards))):
        reward_to_go = gamma*reward_to_go + episode_rewards[i]
        discounted_rewards[i] = reward_to_go
    
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    return tensor(discounted_rewards, dtype=torch.double, device=device)





### model structuring
# see : https://everyday-image-processing.tistory.com/98,
#       https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/12/REINFORCE-CartPole.html
        
class PG:
    def __init__(self, state_shape : int = state_shape, num_actions : int = num_actions):
        self.state_shape, self.num_actions = state_shape, num_actions
        
        self.gamma = gamma
        self.mid_feature = 128
        
        self.episode_log_prob : Tensor = None    # [Tensor(device=cpu), ...]
        self.episode_rewards : list[np.float32] = []    # [float, ...]
        
        self.network = self.build_network()
        # loss function defined manually
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.01)
        
    
    def reset_transition(self):
        self.episode_log_prob = None
        self.episode_rewards = []
    
    
    def build_network(self) -> Sequential:
        """
        Input : state (s_t)
                of size state_shape
        Output : distribution of policy gradient for given state (\pi(a_t|s_t))
                 of size (self.num_actions)  ([probability of choose action 0, ...])
        """
        layer1 = nn.Linear(self.state_shape, self.mid_feature, dtype=torch.double)
        layer2 = nn.Linear(self.mid_feature, num_actions, dtype=torch.double)
        
        model = Sequential(layer1, nn.ReLU(), layer2, nn.Softmax()).to(device)
        return model
    
    
    def select_action(self, state):
        action_dist = self.network(tensor(state, dtype=torch.double, device=device))
        sample_space = Categorical(action_dist)
        action = sample_space.sample()    # a_t
        log_prob = -sample_space.log_prob(action)    # log{\pi(a_t|s_t)}

        return action.item(), log_prob

    
    def store_transition(self, log_prob : Tensor, reward : np.float32):
        if self.episode_log_prob == None:
            self.episode_log_prob = log_prob.unsqueeze(dim=0)
        else: self.episode_log_prob = torch.cat([self.episode_log_prob, log_prob.unsqueeze(dim=0)])
        self.episode_rewards.append(reward)


    def train(self):
        self.network.train()
        
        discounted_reward = discount_and_normalize_rewards(self.episode_rewards)
        log_prob = self.episode_log_prob.to(device=device)
        loss = torch.mul(discounted_reward, log_prob).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.reset_transition()




### training
num_iterations = 1000
pg = PG()
Rewards = []
rewards_sum = 0

print(f"\nChapter 10 : Policy Gradient.    Device : {device}\n")
for i in range(num_iterations):
    done = False
    Return = 0
    state = env.reset()
    
    while not done:
        action, prob = pg.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        pg.store_transition(prob, reward)
        Return += reward
        
        state = next_state
    
    pg.train()
    Rewards.append(Return)
    rewards_sum += Return
    
    if i%20 == 19:
        print(f"Episode {i-18}~{i+1},    Average return {rewards_sum/20:.2f}")
        rewards_sum = 0



graph_obj = go.Scatter(x=[i for i in range(1,num_iterations+1)], y=Rewards)
layout = go.Layout(autosize=True, yaxis=dict(fixedrange=False))
fig = go.Figure(data = [graph_obj], layout=layout)


pio.renderers.default='browser'
fig.show()