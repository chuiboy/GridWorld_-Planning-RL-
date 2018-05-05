import numpy as np
from game import Game

# Define a function to evaluate a policy using Bellman's expectation equation
def policy_evaluation(policy, V, num_eval_iters):
    """
    Returns a vector containing the Value-function for every state following
    the provided policy.
    """

    # Initialize new Value-function to perform synchronous updates
    V_new = np.zeros(policy.shape[0])
    V_new[[3, 6, 13]] = np.nan

    # Evaluate V
    for iter in range(num_eval_iters):
        for state in range(len(V)):

            if state == 0:
                successor_states = [0, 1, 5, 0]
                V_new[state] = -1 + gamma * np.sum(policy[state] * V[successor_states])
            if state == 1:
                successor_states = [1, 2, 1, 0]
                V_new[state] = -1 + gamma * np.sum(policy[state] * V[successor_states])
            if state == 2:
                successor_states = [2, 2, 7, 1]
                V_new[state] = -1 + gamma * np.sum(policy[state] * V[successor_states])
            if state == 4:
                successor_states = [4, 4, 9, 4]
                V_new[state] = -1 + gamma * np.sum(policy[state] * V[successor_states])
            if state == 5:
                successor_states = [0, 5, 10, 5]
                V_new[state] = -1 + gamma * np.sum(policy[state] * V[successor_states])
            if state == 7:
                successor_states = [2, 8, 12, 7]
                V_new[state] = -1 + gamma * np.sum(policy[state] * V[successor_states])
            if state == 8:
                successor_states = [8, 9, 8, 7]
                V_new[state] = -1 + gamma * np.sum(policy[state] * V[successor_states])
            if state == 9:
                successor_states = [4, 9, 14, 8]
                V_new[state] = -1 + gamma * np.sum(policy[state] * V[successor_states])
            if state == 10:
                successor_states = [5, 11, 10, 10]
                V_new[state] = -1 + gamma * np.sum(policy[state] * V[successor_states])
            if state == 11:
                successor_states = [11, 12, 11, 10]
                V_new[state] = -1 + gamma * np.sum(policy[state] * V[successor_states])
            if state == 12:
                successor_states = [7, 12, 12, 11]
                V_new[state] = -1 + gamma * np.sum(policy[state] * V[successor_states])
            if state == 14: # terminal state
                successor_states = [14]
                V_new[state] = 0

        V = np.array(V_new)

    return V

# Define a function to update the policy
def policy_improvement(V):
    """
    Returns the new policy in the form of an array where rows represent the
    states and the columns represent the probability of moving up, right,
    down and left, respectively, given the Value-function.
    """

    # Variable to hold new policy
    policy = np.zeros((len(env.stateSpace), len(env.actionSpace)))
    policy[[3, 6, 13, 14]] = np.nan

    for state in range(len(V)):

        if state == 0:
            successor_states = [0, 1, 5, 0]
            policy[state, np.argmax(V[successor_states])] = 1
        if state == 1:
            successor_states = [1, 2, 1, 0]
            policy[state, np.argmax(V[successor_states])] = 1
        if state == 2:
            successor_states = [2, 2, 7, 1]
            policy[state, np.argmax(V[successor_states])] = 1
        if state == 4:
            successor_states = [4, 4, 9, 4]
            policy[state, np.argmax(V[successor_states])] = 1
        if state == 5:
            successor_states = [0, 5, 10, 5]
            policy[state, np.argmax(V[successor_states])] = 1
        if state == 7:
            successor_states = [2, 8, 12, 7]
            policy[state, np.argmax(V[successor_states])] = 1
        if state == 8:
            successor_states = [8, 9, 8, 7]
            policy[state, np.argmax(V[successor_states])] = 1
        if state == 9:
            successor_states = [4, 9, 14, 8]
            policy[state, np.argmax(V[successor_states])] = 1
        if state == 10:
            successor_states = [5, 11, 10, 10]
            policy[state, np.argmax(V[successor_states])] = 1
        if state == 11:
            successor_states = [11, 12, 11, 10]
            policy[state, np.argmax(V[successor_states])] = 1
        if state == 12:
            successor_states = [7, 12, 12, 11]
            policy[state, np.argmax(V[successor_states])] = 1

    return policy

#############################
###### Policy iteration ######
#############################

# Set parameters
num_iters = 1
num_eval_iters = 7
gamma = 0.8 # discount rate

# Create the environment to extract information about state space and action space
env = Game()

# Initialize policy to select actions uniformly at random
policy = np.ones((len(env.stateSpace), len(env.actionSpace))) / len(env.actionSpace)
policy[[3, 6, 13]] = np.nan

# Initialize Value-function for every state
V = np.zeros(len(env.stateSpace))
V[[3, 6, 13]] = np.nan

# Display initial Value-function and policy
print('Initial Value-function:')
print(V.reshape(3, 5))
print('Initial policy:')
print(policy, '\n')

# Iterate through policy evaluation and policy improvement
for iter in range(num_iters):
    V = policy_evaluation(policy, V, num_eval_iters)
    policy = policy_improvement(V)

    # Display Value-function and policy
    print('##### Iteration {} #####'.format(iter + 1))
    print('Value-function:')
    print(V.reshape(3,5))
    print('Policy:')
    print(policy, '\n')
