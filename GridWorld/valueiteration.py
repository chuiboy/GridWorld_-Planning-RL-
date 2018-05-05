# Value iteration

import numpy as np
from game import Game

# Grid World (states 3, 6, 13 are inaccessible and state 14 is terminal)
gridWorld = np.array(Game().stateSpace)
not_states = [3, 6, 13]
terminal_state = 14

#############################
###### Value iteration ######
#############################

# Initialize Value-function
V = np.zeros(gridWorld.shape)
V[not_states] = np.nan # set inaccessible states to nan
new_V = np.array(V)

# Parameters
gamma = 0.5 # discount rate
num_iters = 10

# Display initialized Value-function
print('Initialized V:')
print(V.reshape(3, 5))

# Perform value iteration
for iter in range(num_iters):

    # Update the value for each state (Bellman's optimality eq.)
    for state in gridWorld:
        if state == 0:
            successor_states = [0, 1, 5]
            new_V[state] = -1 + gamma * np.max(V[successor_states])
        if state == 1:
            successor_states = [0, 1, 2]
            new_V[state] = -1 + gamma * np.max(V[successor_states])
        if state == 2:
            successor_states = [1, 2, 7]
            new_V[state] = -1 + gamma * np.max(V[successor_states])
        if state == 4:
            successor_states = [4, 9]
            new_V[state] = -1 + gamma * np.max(V[successor_states])
        if state == 5:
            successor_states = [0, 5, 10]
            new_V[state] = -1 + gamma * np.max(V[successor_states])
        if state == 7:
            successor_states = [2, 7, 8, 12]
            new_V[state] = -1 + gamma * np.max(V[successor_states])
        if state == 8:
            successor_states = [7, 8, 9]
            new_V[state] = -1 + gamma * np.max(V[successor_states])
        if state == 9:
            successor_states = [4, 8, 9, 14]
            new_V[state] = -1 + gamma * np.max(V[successor_states])
        if state == 10:
            successor_states = [5, 10, 11]
            new_V[state] = -1 + gamma * np.max(V[successor_states])
        if state == 11:
            successor_states = [10, 11, 12]
            new_V[state] = -1 + gamma * np.max(V[successor_states])
        if state == 12:
            successor_states = [7, 11, 12]
            new_V[state] = -1 + gamma * np.max(V[successor_states])
        if state == 14: # terminal state
            successor_states = [14]
            new_V[state] = gamma * np.max(V[successor_states])

    V = np.array(new_V)

    # Display updated Value-function
    print('Iteration {}:'.format(iter + 1))
    print(V.reshape(3, 5))
