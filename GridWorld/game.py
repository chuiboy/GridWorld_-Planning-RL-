import numpy as np

class Game:

    def __init__(self):

        self.stateSpace = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        self.actionSpace = [0, 1, 2, 3]
        self.currentState = 0
        self.goalState = 14
        self.done = 0
        self.reward = 0


    def step(self, action):
        if action == 0: # Up
            if self.currentState not in [0, 1, 2, 8, 4, 11]: # not blocked on top
                self.currentState -= 5 # move up
            self.reward = -1

        elif action == 1: # Right
            if self.currentState not in [2, 4, 5, 9, 12]: # not blocked on right
                self.currentState += 1 # move right
            self.reward = -1

        elif action == 2: # Down
            if self.currentState not in [1, 8, 10, 11, 12]: # not blocked on bottom
                self.currentState += 5 # move down
            self.reward = -1

        elif action == 3: # Left
            if self.currentState not in [0, 4, 5, 7, 10]: # not blocked on left
                self.currentState -= 1 # move left
            self.reward = -1

        else:
            raise ValueError('The requested action is not in actionSpace.')

        if self.currentState == 14: # goal state
            self.done = 1

        return self.currentState, self.reward, self.done

    def reset(self):
        self.currentState = 0
        self.done = 0

    def playGame(self, Q):

        def displayGrid(state):
            for i in range(15):
                if i == state:
                    if state in [4, 9, 14]:
                        print('A')
                    else:
                        print('A', end='')
                elif i in [3, 6, 13]:
                    print('X', end='')
                elif i == 14:
                    print('G')
                elif i in [4, 9]:
                    print('O')
                else:
                    print('O', end='')

        self.reset()
        state = self.currentState
        done = 0

        print('A = agent, X = blocked, G = goal, O = open')
        print('\ntimestep 0 (start):')
        displayGrid(state)

        t = 1
        while done == 0:
            print('\ntimestep {}:'.format(t))
            action = np.argmax(Q[state])
            state, _, done = self.step(action)
            displayGrid(state)
            t += 1
