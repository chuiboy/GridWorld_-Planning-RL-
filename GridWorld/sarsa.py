# Sarsa(lambda) learning

# Import dependencies
import numpy as np
import random
from game import Game

# Create environment
env = Game()

# Primary parameters
num_eps = 1000
T = 1000 # horizon
discount_rate = 0.99
learning_rate = 0.1
decay_rate = 0.8

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

    # Initialize eligibility trace for all s, a
    E = np.zeros((len(env.stateSpace), len(env.actionSpace)))

    # Initial action according to policy
    if random.uniform(0, 1) > epsilon:
        action = np.argmax(Q[env.currentState])
    else:
        action = random.choice(env.actionSpace)

    # Sample episode
    for t in range(T):

        # Take a step in the environment
        next_state, reward, done = env.step(action)

        # Select next action according to policy
        if random.uniform(0, 1) > epsilon:
            next_action = np.argmax(Q[env.currentState])
        else:
            next_action = random.choice(env.actionSpace)

        # Calculate the TD error
        td_error = reward + discount_rate * Q[next_state, next_action] - Q[state, action]

        # Update the eligibility trace for the current (s,a) pair
        E[state, action] += 1

        # Update action-value function and eligibility trace for all (s,a) pairs
        for s in env.stateSpace:
            for a in env.actionSpace:
                Q[s, a] += learning_rate * td_error * E[s, a]
                E[s, a] = discount_rate * decay_rate * E[s, a]

        state = next_state
        action = next_action

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
