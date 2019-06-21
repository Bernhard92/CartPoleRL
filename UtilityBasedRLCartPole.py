#!/usr/bin/env python
# coding: utf-8

# # Active Greedy Utility-based Agent

# In[2]:


import gym 
import operator
import itertools
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import deque
from random import randint


# In[3]:


env = gym.make('CartPole-v1')
env._max_episode_steps = 5000
ACTION_SPACE = env.action_space.n # number of possible actions
OBSERVATION_SPACE = env.observation_space.shape[0] # number of observable variables
EXPLORATION_PROB = 1.0 # EPSILON GREEDY STRATEGY
EXPLORATION_DECAY = 0.995
EXPLORATION_STOP = 0.001


# In[4]:


def create_state_intervals():
    intervals = np.zeros((OBSERVATION_SPACE, STATES_IN_INTERVAL))
    intervals[0] = np.linspace(-4.8, 4.8, STATES_IN_INTERVAL)
    intervals[1] = np.linspace(-3.5, 3.5, STATES_IN_INTERVAL)
    intervals[2] = np.linspace(-0.42, 0.42, STATES_IN_INTERVAL)
    intervals[3] = np.linspace(-4, 4, STATES_IN_INTERVAL)
    return intervals


# In[5]:


def discretize_observation(observation):
    discrete_observation = np.array([np.digitize(observation[index], INTERVALS[index])-1 for index in range(OBSERVATION_SPACE)])
    # if some value is under the lower border ignore it and give it min value
    discrete_observation = [0 if x<0 else x for x in discrete_observation]
    return discrete_observation


# In[6]:


def get_all_possible_states():
    digits = len(str(STATES_IN_INTERVAL))
    state_indices = [str(state_index).zfill(digits) for state_index in range(STATES_IN_INTERVAL)] # all encodings for a single observation variable
    states = [state_indices for i in range(OBSERVATION_SPACE)] # for each observation variable a list of its encodings
    states = list(itertools.product(*states)) # get all permutation of all state encodings (->list of tuples)
    states = [''.join(x) for x in states] # join tuples to a single string
    return states


# In[7]:


def observation_to_state(observation):
    discrete_observation = discretize_observation(observation)
    digits = len(str(STATES_IN_INTERVAL))
    
    state = ''
    for state_id in discrete_observation:
        if len(str(state_id)) < digits:
            state += str(state_id).zfill(digits)
        else:
            state += str(state_id)
    return state


# In[8]:


env = gym.make('CartPole-v1')
env._max_episode_steps = 5000
ACTION_SPACE = env.action_space.n #number of possible actions
OBSERVATION_SPACE = env.observation_space.shape[0] #number of observable variables
STATES_IN_INTERVAL = 16
#LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EXPLORATION_PROB = 1.0 # EPSILON GREEDY STRATEGY
EXPLORATION_DECAY = 0.995
EXPLORATION_STOP = 0.01


# In[9]:


def create_utility_table():
    states = get_all_possible_states()
    utility_values = np.zeros(len(states))
    utility_table = dict(zip(states, utility_values))
    return utility_table


# In[10]:


def create_reward_table():
    states = get_all_possible_states()
    rewards = np.zeros(len(states)) # init with zero; high; random
    reward_table = dict(zip(states, rewards))
    return reward_table


# In[11]:


#create_utility_table()


# In[12]:


# def update_number_state_action_table(nsa_table, state, action):
#     key = (state, action)
#     if key in nsa_table.keys():
#         nsa_table[key] += 1
#     else:
#         nsa_table[key] = 1


# In[13]:


def update_number_state_action_next_state_table(nsas_table, state, action, next_state):
    if (state, action) in nsas_table.keys():
        if next_state in nsas_table[(state, action)].keys():
            nsas_table[(state, action)][next_state] += 1
        else:
            nsas_table[(state, action)][next_state] = 1
    else:
        nsas_table[(state, action)] = {}
        nsas_table[(state, action)][next_state] = 1
        
    return nsas_table


# In[14]:


def get_transition_probability(nsas_table, state, action):
    next_states = nsas_table[(state, action)]
    temp = {}
    for next_state in next_states:
        temp[next_state] = nsas_table[(state, action)][next_state]/sum(nsas_table[(state, action)].values())
        
    return temp


# In[15]:


def get_nsa(nsas_table, state, action):
    return sum(nsas_table[(state, action)].values())


# In[16]:


#def get_utility(utility_table, ) ## maybe later


# In[17]:


def update_utility_estimate(utility_table, nsas_table, state, action, next_state, reward_table, epsilon, gamma):
    next_states = nsas_table[(state, action)].keys()
    u = 0
    probs = get_transition_probability(nsas_table, state, action)
    for next_state in next_states:
        u +=  probs[next_state] * utility_table[next_state] ##
    
    actions = [0, 1]
    f_values = []
    if (state, actions[0]) in nsas_table.keys():
        f_values.append(exploration_function(u, get_nsa(nsas_table, state, action), epsilon))
    if (state, actions[1]) in nsas_table.keys():
        f_values.append(exploration_function(u, get_nsa(nsas_table, state, action), epsilon))
    if not f_values:
        print('(O.O) we have a problem')
    
    utility_table[state] = reward_table[state] + gamma * max(f_values)
    return utility_table


# In[18]:


def exploration_function(utility, n, epsilon):
    if n < epsilon:
        return 100
    else:
        return utility


# In[19]:


def get_action(utility_table, nsas_table, state): 
    best_a0, best_a1 = -100, -100
    if (state, 0) in nsas_table.keys():
        next_states_a0 = nsas_table[(state, 0)].keys()
        best_a0 = max([utility_table[s] for s in next_states_a0])
    if (state, 1) in nsas_table.keys():
        next_states_a1 = nsas_table[(state, 1)].keys()
        best_a1 = max([utility_table[s] for s in next_states_a1])
    
    if best_a0 == best_a1:
        return randint(0,1)
    
    return (0 if best_a0 > best_a1 else 1)


# In[21]:


INTERVALS = create_state_intervals()




number_of_games = 1000


#data = pd.DataFrame(columns = ['mean', 'max', 'solve_step'])
data_max = []
data_mean = []
data_solve = []

solved = False

for i in range(10):
    print('EPISODE:', i)
        # init things we dont understand
    utility_table = create_utility_table()
    nsas_table = {}
    reward_table = create_reward_table()

    gamma = 0.99
    explore = 5 ## change this (jahrers idea ... critical)
    epsilon = EXPLORATION_PROB
    
    if i != 0:
        data_max.append(game_max)
        data_mean.append(game_mean)
        if not solved:
            data_solve.append(-1)
    last100_rewards = deque(maxlen=100) # fifo queue
    game_max = []
    game_mean = []
    solved = False
    for game in range(number_of_games):
        overall_reward, done = 0, False
        observation = env.reset()
        state = observation_to_state(observation)

        while not done:
            if game % 100 == 0: env.render()
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = get_action(utility_table, nsas_table, state)

            observation, reward, done, _ = env.step(action)
            next_state = observation_to_state(observation)
            overall_reward += reward

            if done: reward = -500 # punish if agent dies

            reward_table[state] = reward
            nsas_table = update_number_state_action_next_state_table(nsas_table, state, action, next_state)

            utility_table = update_utility_estimate(utility_table, nsas_table, state, action, next_state, reward_table, explore, gamma)

            state = next_state

        if epsilon > EXPLORATION_STOP: epsilon *= EXPLORATION_DECAY

        if game % 100 == 0 and game != 0:
            print('Episode:', game, 'Epsilon:', round(epsilon, 3), 
                  'Mean-Reward:', np.mean(last100_rewards), 'Max-Reward:', max(last100_rewards))
        if (np.mean(last100_rewards) >= 195) and not solved: 
            print('TASK COMPLETED LAST 100 GAMES HAD AN AVERAGE SCORE >=195 ON GAME', game)
            print(last100_rewards)
            solved = True
            data_solve.append(game)
            
        if game % 100 == 0 and game != 0:
            game_max.append(max(last100_rewards))
            game_mean.append(np.mean(last100_rewards))
        
        last100_rewards.append(overall_reward) 
        


# In[36]:


stat.head()


# In[37]:


stat.to_csv('testrun.csv')

