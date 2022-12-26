import gym

env = gym.make('FrozenLake-v0')

### basic functioning
env.render()
print(env.observation_space)
print(env.action_space)

print(env.P[0][2])
print(env.P[6][0])
print(env.P[5][4])    # state 'H' -> no more action is available, so printed result is [(1.0, 5, 0, True)].



### generating episode

state = env.reset()
num_episodes = 10
num_timesteps = 20


for i in range(num_episodes):
    state = env.reset()
    print(f'========== Episode {i+1} ==========')
    print('Time Step 0 : '); env.render()
    for t in range(num_timesteps):
        random_action = env.action_space.sample()
        next_state, reward, done, info = env.step(random_action)
        print(f'Time Step {t+1} : '); env.render()
        if done: break
    print('')