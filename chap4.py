import random
import gym
import pandas as pd
from collections import defaultdict

env = gym.make('Blackjack-v0')

print(env.reset())
print(env.action_space)


def policy(state):
    return 0 if state[0]>19 else 1

### single trial
num_timesteps = 100
def generate_episode(policy):
    episode = []
    state = env.reset()

    for _ in range(num_timesteps):
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        if done: break
        state = next_state
    
    return episode

generate_episode(policy)



### multi trial - computing value function
## every-visit Monte Carlo
total_return = defaultdict(float)
N = defaultdict(int)
num_iterations = 500000

for i in range(num_iterations):
    episode = generate_episode(policy)
    states, actions, rewards = zip(*episode)
    for t, state in enumerate(states):
        total_return[state] += (sum(rewards[t:]))
        N[state] += 1

total_return = pd.DataFrame(total_return.items(), columns = ['state', 'total_return'])
N = pd.DataFrame(N.items(), columns=['state','N'])
df = pd.merge(total_return, N ,on='state')
df['value'] = df['total_return']/df['N']

df.head(10)
df[df['state'] == (21,9,False)]['value'].values

## first-visit Mote Carlo
total_return = defaultdict(float)
N = defaultdict(int)
num_iterations = 500000

for i in range(num_iterations):
    episode = generate_episode(policy)
    states, actions, rewards = zip(*episode)
    for t, state in enumerate(states):
        if state not in states[0:t]:
            total_return[state] += (sum(rewards[t:]))
            N[state] += 1

total_return = pd.DataFrame(total_return.items(), columns = ['state', 'total_return'])
N = pd.DataFrame(N.items(), columns=['state','N'])
df = pd.merge(total_return, N ,on='state')
df['value'] = df['total_return']/df['N']

df.head(10)
df[df['state'] == (21,9,False)]['value'].values



### on-policy epsilon-greedy MC control (by first-visit)
Q = defaultdict(float)
total_return = defaultdict(float)
N = defaultdict(int)
epsilon = 0.5

def epsilon_greedy_policy(state, Q):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x: Q[(state,x)])

def generate_episode(Q):
    episode = []
    state = env.reset()

    for _ in range(num_timesteps):
        action = epsilon_greedy_policy(state, Q)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
    
    return episode

for _ in range(num_iterations):
    episode = generate_episode(Q)    # because in first-visit, every Q[state-action pair] updated at most once
    all_state_action_pairs = [(s,a) for (s,a,r) in episode]
    rewards = [r for (s,a,r) in episode]
    
    for t, (state,action,_) in enumerate(episode):
        if (state,action) in all_state_action_pairs[0:t]: continue
        R = sum(rewards[t:])
        total_return[(state,action)] += R
        N[(state,action)] += 1
        Q[(state,action)] = total_return[(state,action)] / N[(state,action)]

df_on = pd.DataFrame(Q.items(), columns=['state_action pair', 'value'])



### off-policy MC control (by every-visit)
Q_ = defaultdict(float)
total_return_ = defaultdict(float)
cumulative_weight_ = defaultdict(float)
epsilon = 0.5

target_env = gym.make('Blackjack-v0')

# def epsilon_greedy_policy(state, Q=Q):
#     if random.uniform(0,1) < epsilon:
#         return env.action_space.sample()
#     else:
#         return max(list(range(env.action_space.n)), key = lambda x: Q[(state,x)])

def greedy_policy(state, Q=Q_):
    return max(list(range(target_env.action_space.n)), key = lambda x: Q[(state,x)])

def generate_episode(Q=Q):
    episode = []
    state = target_env.reset()

    for _ in range(num_timesteps):
        action = epsilon_greedy_policy(state, Q)
        next_state, reward, done, info = target_env.step(action)
        episode.append((state, action, reward))
    
    return episode

for _ in range(num_iterations):
    episode = generate_episode(Q); episode.reverse()    # because Q does not updated
    W = 1

    for t, (state,action,reward) in enumerate(episode):
        prev_greedy_target_policy = greedy_policy(state)
        total_return_[(state,action)] += reward
        cumulative_weight_[(state,action)] += W
        Q_[(state,action)] += W/cumulative_weight_[(state,action)] * (total_return_[(state,action)] - Q_[(state,action)])
        if greedy_policy(state) != prev_greedy_target_policy: break    # \pi(a|s) = 0 -> W *= 0 -> no update anymore
        else: W /= (1-epsilon)    # \pi(a|s) = 1, b(a|s) = 1-\epsilon when a is maximal, by epsilon-greedy policy

df_off = pd.DataFrame(Q_.items(), columns=['state_action pair', 'value'])
