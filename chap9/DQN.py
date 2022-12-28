import random
from collections import deque

import numpy as np
import gym

import torch
from torch import tensor
from torch import nn, optim, Tensor
from torch.nn import Sequential

cpu = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"





### setting
env = gym.make('MsPacman-v0')
state_size = (88,80,1)
action_size = env.action_space.n


color = np.array([210,164,74]).mean()

def preprocess_state(state):
    image = state[1:176:2, ::2]    # crop and resize : 210x160x3 -> 176x160x3 -> 88x80x3
    image = image.mean(axis=2)    # grayscale : 88x80x3 -> 88x80
    image[image==color] = 0    # improve image contrast
    image = (image-128)/128 - 1    # normalize
    image = np.expand_dims(image.reshape(88, 80), axis=0)    # reshape
    return image

total_training_num, errored_training_num, done_training_num = 0,0,0

### Deep Q-Learning (DQN)

## DQN class
class DQN:
    def __init__(self, state_size, action_size : int):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=10000)
        
        self.gamma = 0.9
        self.epsilon = 0.8
        self.update_rate = 1000
        
        
        self.main_network = self.build_network()
        self.main_loss = nn.MSELoss()
        self.main_optimizer = optim.Adam(self.main_network.parameters(), lr=1e-07)
        
        self.target_network = self.build_network()
        self.target_loss = nn.MSELoss()
        self.target_optimizer = optim.Adam(self.main_network.parameters(), lr=1e-07)
        
        self.update_target_network()
    
    
    def build_network(self) -> Sequential:
        """
        Input : 2d picture image
                of size (88,80)
        Output : expected Q-value(or rewards) of each action
                 of size (self.action_size)  ([Q-value of action 0, ... Q-value of last action])
                 Highest value -> chosen action
        """
        layer1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0, device=device, dtype=torch.float)
        layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=(1,1), device=device, dtype=torch.float)
        layer3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0, device=device, dtype=torch.float)
        
        layer4 = nn.LazyLinear(512, device=device, dtype=torch.float)
        layer5 = nn.LazyLinear(self.action_size, device=device, dtype=torch.float)
        
        
        model = Sequential(layer1, nn.ReLU(), layer2, nn.ReLU(), layer3, nn.ReLU(), nn.Flatten(), layer4, nn.ReLU(), layer5).to(device)
        
        return model
    
    
    def store_transition(self, state : np.float32, action, reward : np.float32, next_state : np.float32, done : bool):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    
    def epsilon_greedy(self, state):
        if random.uniform(0,1) < self.epsilon: return np.random.randint(self.action_size)
        Q_values = self.main_network(state).clone().detach().to(device=cpu)
        return np.argmax(Q_values[0]).item()
    
    
    def train(self, pred_Q : Tensor, target_Q : Tensor, model : Sequential, loss_fn, optimizer):
        # compute loss
        loss : Tensor = loss_fn(pred_Q, target_Q)
        
        # backpropagation
        optimizer.zero_grad()    # reset gradient
        loss.backward()    # save gradiant based on loss
        optimizer.step()    # update parameters(weight)
    
    
    def train_model(self, batch_size):
        self.main_network.train()
        self.target_network.eval()
        
        minibatch = random.sample(self.replay_buffer, batch_size)
        
        done_data = np.array([datum[4] for datum in minibatch])
        done_index = np.where(done_data == True)[0]
        not_done_index = np.setdiff1d( np.arange(done_data.size), done_index )
        
        state_data = tensor([datum[2] for datum in minibatch], dtype=torch.float, device=device)
        reward_data = tensor([datum[2] for datum in minibatch], dtype=torch.float, device=device)
        next_state_data = tensor([datum[3] for datum in minibatch], dtype=torch.float, device=device)
        
        # pred_Q / target_Q. Shuffle if needed
        non_final_Q =( (self.main_network(state_data[ind]).max(), reward_data[ind]+self.gamma*self.target_network(next_state_data[ind]).max()) for ind in not_done_index )
        final_Q = ()
        if done_index.size > 0: final_Q = ( (self.main_network(state_data[ind]).max(), reward_data[ind]) for ind in done_index)
        
        # train
        self.main_network.train()
        for pred_Q, target_Q in non_final_Q:
            self.train(pred_Q, target_Q, self.main_network, self.main_loss, self.main_optimizer)
        
        if done_index.size > 0:
            # pass reward_data[ind] has size torch.Size([]) -> exclude from training
            
            global done_training_num
            done_training_num += done_index.size
            for pred_Q, target_Q in final_Q:
                self.train(pred_Q, target_Q, self.main_network, self.main_loss, self.main_optimizer)
        
        del minibatch, final_Q, non_final_Q
    
    
    # # print test_loss or correct to visualize correctness
    # def test(self, state, target_Q, model, loss_fn):
    #     # set evalunation mode
    #     model.eval()
    #     with torch.no_grad():
    #         X, y = state.to(device), target_Q.to(device)
    #         test_loss, correct = 0, 0
            
    #         pred_Q = model(X)
    #         test_loss += loss_fn(pred_Q, y).item()
    #         correct += (pred_Q.argmax(1) == y).type(torch.float).sum().item()
    
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())




## training
num_episodes = 500
num_timesteps = 20000
batch_size = 20
num_screens = 4
done = False
time_step = 0
# epoch = 1 in here

print(f"\nChapter 9 : DQN.    Device : {device}\n")
dqn = DQN(state_size, action_size)
for i in range(num_episodes):
    Return = 0
    state = preprocess_state(env.reset(), device=device)
    for t in range(num_timesteps):
        
        # env.render()
        time_step += 1
        if time_step%dqn.update_rate == 0:
            dqn.update_target_network()
        
        # create training set
        action = dqn.epsilon_greedy(state)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(next_state, device=device)
        dqn.store_transition(state, action, reward, next_state, done)
        
        state = next_state
        Return += reward
        if done: print(f"Episode {i+1}, Return {Return:.1f}. error/done/total = {errored_training_num}/{done_training_num}/{total_training_num}"); break
        # training when enoughly many training set is ready
        if len(dqn.replay_buffer) > batch_size: dqn.train_model(batch_size)