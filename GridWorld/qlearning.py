# Q-Learning

import numpy as np
import random
from game import Game

env = Game()
print('state space:', env.stateSpace)
print('action space:', env.actionSpace)

######################
##### Q-Learning #####
######################

# Parameters
num_eps = 100
gamma = 0.9 # discount rate
learning_rate = 0.5

# Parameters for dynamic exploration rate
epsilon = 1.0 # exploration rate
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.03

# Initialize Q-table
qtable = np.zeros((len(env.stateSpace), len(env.actionSpace)))

# Run for many episodes
for ep in range(num_eps):

    env.reset()
    total_reward = 0

    # Single episode (from start to goal)
    while True:

        # Select action
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(qtable[env.currentState, :])
        else:
            action = random.choice(env.actionSpace)

        # Perform action and update Q-table
        state = env.currentState
        new_state, reward, done = env.step(action)
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        # Accumulated reward for episode
        total_reward += reward

        # Break out of episode once goal reached
        if done == 1:
            print('total reward for episode {}: {}'.format(ep, total_reward))
            break

    # Update exploration rate for the next episode
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*ep)

# ​Final Q-table
print(qtable)


##############################
##### Agent under policy #####
##############################

# Function to display Grid World given agent's state
def displayGrid(state):
    print('A = agent, X = blocked, G = goal, O = open')
    for i in range(15):
        if i == state:
            if state in [4, 9, 14]:
                print('A')
            else:
                print('A', end='')
        elif i == 0:
            print('S', end='')
        elif i in [3, 6, 13]:
            print('X', end='')
        elif i == 14:
            print('G')
        elif i in [4, 9]:
            print('O')
        else:
            print('O', end='')

# Run agent through Grid World using learnt policy
env.reset()
total_reward = 0
displayGrid(env.currentState)

while True:

    # Select action
    action = np.argmax(qtable[env.currentState, :])

    # Perform action
    state = env.currentState
    new_state, reward, done = env.step(action)

    # Display updated Grid World
    displayGrid(new_state)

    # Accumulated reward
    total_reward += reward

    # Break out of episode once goal reached
    if done == 1:
        print('total reward:', total_reward)
        break
