import warnings
warnings.filterwarnings('ignore')

import numpy as np
import gym
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
tf = tf.compat.v1




### create environment
env = gym.make("Pendulum-v0").unwrapped

state_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]
action_bound = [env.action_space.low, env.action_space.high]

# variables
epsilon = 0.2


### PPO class
class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.state_ph = tf.placeholder(tf.float32, [None, state_shape], 'state')
        
        # value network
        with tf.variable_scope('value'):
            layer1 = tf.layers.dense(self.state_ph, 100, tf.nn.relu)
            self.v = tf.layers.dense(layer1, 1)
            self.Q = tf.placeholder(tf.float32, [None,1], 'discounted_r')
            self.advantage = self.Q - self.v
            self.value_loss = tf.reduce_mean(tf.square(self.advantage))
            self.train_value_nw = tf.train.AdamOptimizer(0.002).minimize(self.value_loss)
        
        # policy network
        pi, pi_params = self.build_policy_network('pi', trainable=True)
        oldpi, oldpi_params = self.build_policy_network('oldpi', trainable=False)

        with tf.variable_scope('sameple_action'):    # sampling
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)
        with tf.variable_scope('update_oldpi'):    # update parameters of old policy
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        # objective function
        self.action_ph = tf.placeholder(tf.float32, [None, action_shape], 'action')
        self.advantage_ph = tf.placeholder(tf.float32, [None,1], 'advantage')

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.action_ph)/oldpi.prob(self.action_ph)
                objective = ratio*self.advantage_ph
                L = tf.reduce_mean( tf.minimum(objective, tf.clip_by_value(ratio, 1-epsilon, 1+epsilon)*self.advantage_ph) )    
            self.policy_loss = -L
        
        with tf.variable_scope('train_policy'):
            self.train_policy_nw = tf.train.AdamOptimizer(0.001).minimize(self.policy_loss)
        
        self.sess.run(tf.global_variables_initializer())

    
    ## train function
    def train(self, state, action, reward):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.state_ph : state, self.Q : reward})
        # training
        [self.sess.run(self.train_policy_nw, {self.state_ph : state, self.action_ph : action, self.advantage_ph : adv}) for _ in range(10)]
        [self.sess.run(self.train_value_nw, {self.state_ph : state, self.Q : reward}) for _ in range(10)]
    

    ## build policy network
    def build_policy_network(self, name, trainable):
        with tf.variable_scope(name):
            layer = tf.layers.dense(self.state_ph, 100, tf.nn.relu, trainable=trainable)
            mu = 2*tf.layers.dense(layer, action_shape, tf.nn.tanh, trainable=trainable)    # mean
            sigma = tf.layers.dense(layer, action_shape, tf.nn.softplus, trainable=trainable)    # std

            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params


    ## select action
    def select_action(self, state):
        state = state[np.newaxis, :]
        action = self.sess.run(self.sample_op, {self.state_ph:state})[0]
        action = np.clip(action, action_bound[0], action_bound[1])
        return action
    

    ## compute value network
    def get_state_value(self, state):
        if state.ndim < 2: state = state[np.newaxis, :]
        return self.sess.run(self.v, {self.state_ph: state})[0,0]



### training PPO network
ppo = PPO()
num_episodes = 1000
num_timesteps = 200
gamma = 0.9
batch_size = 32

for i in range(num_episodes):
    state = env.reset()
    episode_states, episode_actions, episode_rewards = [], [], []
    Return = 0

    for t in range(num_timesteps):
        env.render()
        action = ppo.select_action(state)
        next_state, reward, done, _ = env.step(action)

        episode_states.append(state); episode_actions.append(action); episode_rewards.append((reward+8)/8)
        state = next_state
        Return += reward

        # update network when number of batch_size data is obtained
        if (t+1%batch_size) == 0 or t == num_timesteps-1:
            v_s_ = ppo.get_state_value(next_state)

            discounted_r = []
            for reward in episode_rewards[::-1]:
                v_s_ = reward + gamma*v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            es, ea, er = np.vstack(episode_states), np.vstack(episode_actions), np.vstack(discounted_r)[:, np.newaxis]
            ppo.train(es,ea,er)

            episode_states, episode_actions, episode_rewards = [], [], []
        
    if i%10 == 0: print(f"Episode: f{i}, Return: f{Return}")