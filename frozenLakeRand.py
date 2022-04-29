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


for _ in range(1000):                
    random_action = env.action_space.sample()
    new_state, reward, done, info = env.step(random_action)
    # print(random_action)
    print(f'Reward = {reward}')
    env.render()
    if done:
        break