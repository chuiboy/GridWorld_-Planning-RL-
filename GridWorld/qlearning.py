# Q-learning

# Import dependencies
import numpy as np
from game import Game
import random

# Create environment
env = Game()

# Primary parameters
num_eps = 1000
T = 1000 # horizon
discount_rate = 0.99
learning_rate = 0.1

# Parameters for epsilon-greedy policy improvement
epsilon = 1.0 # exploration rate
epsilon_decay_rate = 0.003

# Initialize Q(s,a) for all s, a
Q = np.zeros((len(env.stateSpace), len(env.actionSpace))) # action-value function for all (s,a) pairs

# Run "num_eps" episodes
for ep in range(num_eps):

    # Reset the environment and initialize start state
    env.reset()
    state = env.currentState

    # Sample episode
    for t in range(T):

        # Select next action according to policy
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(Q[env.currentState])
        else:
            action = random.choice(env.actionSpace)

        # Take a step in the environment
        next_state, reward, done = env.step(action)

        # Update action-value function for current (s,a) pairs
        Q[state, action] += learning_rate * (reward + discount_rate * Q[next_state].max() - Q[state, action])

        state = next_state

        # End episode once terminal state reached
        if done:
            break

        # Update exploration rate
        epsilon = np.exp(-epsilon_decay_rate * ep)

# Display the action-value function
print('Action-Value function for all states and actions:')
print(Q)

# Run through the grid world following the learnt policy
env.playGame(Q)
