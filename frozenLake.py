import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import time 

# actions = {
#     'Left': 0,
#     'Down': 1,
#     'Right': 2, 
#     'Up': 3
# }
 
# print('---- winning sequence ------ ')
# winning_sequence = (2 * ['Right']) + (3 * ['Down']) + ['Right']
# print(winning_sequence)

env = gym.make("FrozenLake-v1", is_slippery=False)
env.reset()
env.render()

alpha = 0.5
gamma = 0.9
outcomes = []
sequence = []

qtable = np.zeros((env.observation_space.n, env.action_space.n))

print('Q-table =')
print(qtable)


for _ in range(10000):
    state = env.reset()
    done = False

    outcomes.append("Failure")

    while not done:
        if np.max(qtable[state]) > 0:
          action = np.argmax(qtable[state])
        else:
          action = env.action_space.sample()

        sequence.append(action)
             
        new_state, reward, done, info = env.step(action)

        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        
        state = new_state

        env.render()
        time.sleep(0.1)

        if reward:
          outcomes[-1] = "Success"
print(f"Sequence = {sequence}")
print()
print('===========================================')
print('Q-table after training:')
print(qtable)
