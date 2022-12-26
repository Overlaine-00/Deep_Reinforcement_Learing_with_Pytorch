import random
import gym
import pandas as pd

env = gym.make('FrozenLake-v0')

def random_policy():
    return env.action_space.sample()

alpha = 0.85    # learning rate
gamma = 0.9    # discount factor
num_episode, num_timesteps = 50000, 1000



### TD prediction basic (TD is (by definition) every-visit)
V = {}
for s in range(env.observation_space.n): V[s] = 0.0

for i in range(num_episode):
    s = env.reset()
    for t in range(num_timesteps):
        a = random_policy()
        s_, r, done, _ = env.step(a)
        V[s] += alpha*(r + gamma*V[s_] - V[s])
        s = s_
        if done: break

df = pd.DataFrame(list(V.items()), columns = ['state', 'value'])



### on-policy method : SARSA learning
epsilon = 0.8
Q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s,a)] = 0.0


def epsilon_greedy(state, epsilon = epsilon):
    if random.uniform(0,1) < epsilon: return env.action_space.sample()
    else: return max(list(range(env.action_space.n)), key=lambda x: Q[(state,x)])

for _ in range(num_episode):
    s = env.reset()
    a = epsilon_greedy(s)

    for t in range(num_timesteps):
        s_, r, done, __ = env.step(a)
        a_ = epsilon_greedy(s_)
        Q[(s,a)] += alpha*(r+gamma*Q[(s_,a_)] - Q[(s,a)])
        s,a = s_,a_
        if done: break



### off-policy method : Q learning
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s,a)] = 0.0

for _ in range(num_episode):
    s = env.reset()
    for t in range(num_timesteps):
        a = epsilon_greedy(s,epsilon)
        s_, r, done, __ = env.step(a)
        greedy_Q_val = max(range(env.action_space.n), key=lambda x: Q[(s_,x)])    # to get greedy action a_, use np.argmax
        Q[(s,a)] += alpha*(r + gamma*(Q[(s_,a_)]) - Q[(s,a)])
        s = s_
        if done: break
