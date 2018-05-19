# Monte-Carlo Learning (First-Visit)

# Import dependencies
import numpy as np
from game import Game
import random

# Create the environment
env = Game()

# Primary parameters
num_eps = 1000
T = 1000 # horizon
discount_rate = 0.99

# Parameters for epsilon-greedy policy improvement
epsilon = 1.0 # exploration rate
epsilon_decay_rate = 0.001

# Initialize Q(s,a) and N(s,a) for all s, a
Q = np.zeros((len(env.stateSpace), len(env.actionSpace))) # action-value function for all (s,a) pairs
N = np.zeros((len(env.stateSpace), len(env.actionSpace))) # number of first-visit occurrences for all (s,a) pairs

# Run "num_eps" episodes
for ep in range(num_eps):

    # Reset the environment and initialize start state
    env.reset()
    state = env.currentState

    # Initialize arrays that will be used later on
    discounted_return = np.zeros((len(env.stateSpace), len(env.actionSpace))) # discounted return for each (s,a) pair
    first_visit = np.ones((len(env.stateSpace), len(env.actionSpace))) * -1 # timestep of first-visit occurrence for each (s,a) pair

    # Sample episode
    for t in range(T):

        # Take action according to policy
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(Q[env.currentState])
        else:
            action = random.choice(env.actionSpace)

        # Update first_visit(s,a) and N(s,a) for current (s,a) pair
        if first_visit[state, action] == -1:
            first_visit[state, action] = t
            N[state, action] += 1

        # Take a step in the environment
        state, reward, done = env.step(action)

        # Update discounted return for all (s,a) pairs
        discounted_return += (first_visit != -1) * (discount_rate**(t-first_visit) * reward)

        # End episode once terminal state reached
        if done:
            break

    # Update action-value function
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            if first_visit[i,j] != -1:
                Q[i,j] += (1/N[i,j]) * (discounted_return[i,j] - Q[i,j])

    # Update exploration rate
    epsilon = np.exp(-epsilon_decay_rate * ep)

# Display the action-value function
print('Action-Value function for all states and actions:')
print(Q)

# Run through the grid world following the learnt policy
env.playGame(Q)
