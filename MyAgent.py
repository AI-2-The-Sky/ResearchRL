import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
import numpy as np
import random

from bomberman.agents.BaseAgent   import BaseAgent
from bomberman.states.State       import State
from bomberman.defines            import t_action
from bomberman                    import defines

class NeuralNetwork(nn.Module):
    def __init__(self, nb_state, nb_action):
        super().__init__()
        # fc => fully connected
        # each one represents a layer
        self.fc1 = nn.Linear(nb_state, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, nb_action)
    
    # the forward function define the computation
    # performed at every call
    def forward(self, x):
        x = self.fc1(x)
        # I didn't quit understand the use of the relu function
        # For future me: please understand why !
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class MyAgent(BaseAgent):
    def __init__(self,
            player_num,
            nb_episode=100,
            nb_state=(11*11)+6,                             # the number of different state variables given to the agent
                                                            # Here there is 127 differents state 11x11 map + 6 characteristics states
            nb_action=len(defines.move_space),              # the number of different actions that can choose the agent
            batch_size=24,
            memory_size=3000,                               # memory that store the state that will be replayed
            alpha=0.01,                                     # aka learning rate
            gamma=1.0,                                      # aka discount factor
            epsilon=1.0,                                    # aka exploitation vs exploration factor
            epsilon_min=0.01,
            epsilon_decay=0.995):
        super().__init__(player_num)
        self.nb_episode = nb_episode
        self.nb_state = nb_state
        self.nb_action = nb_action
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.last_p_state = None
        self.last_action = None

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.neural_network = NeuralNetwork(self.nb_state, self.nb_action).to(self.device)
        self.loss_function = nn.MSELoss()
        # the NN optimizer (didn't understand it at all (you know it's for you, future me))
        self.optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=self.alpha)

        def decreaseEpsilon(self):
            if self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        def choose_action(self, state):
            # exploration vs exploitation
            if np.random.random() <= self.epsilon:
                return np.random.randint(0, nb_action)
            else:
                with torch.no_grad():
                    return torch.argmax(self.neural_network(state)).numpy
            
            self.decreaseEpsilon()
        
        # remember actual state variables for replay
        def rememberState(self, state, action, reward, next_state, done):
            reward = torch.tensor(reward)
            self.memory.append((state, action, reward, next_state, done))
        
        # replay memory to evaluate and modify neuralNetwork
        def replay(self):
            y_batch, y_target_batch = [], []
            minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

            for state, action, reward, next_state, done in minibatch:
                y = self.neural_network(state)
                y_target = y.clone().detach()
                with torch.no_grad():
                    y_target[0][action] = reward if done else reward + self.gamma * torch.max(self.neural_network(next_state)[0])
                y_batch.append(y[0])
                y_target_batch.append(y_target[0])
        
            y_batch = torch.cat(y_batch)
            y_target_batch = torch.cat(y_target_batch)

            self.optimizer.zero_grad()
            loss = self.loss_function(y_batch, y_target_batch)
            loss.backward()
            self.optimizer.step()

        def preprocessState(self, rawState):
            # extract the State variables from rawState
            # with:
            #   map as a 11x11 list that represents the board
            #   playerState a structure that regroups multiple status variable
            #   winner an int that give the winner Id (if there is one)
            (map, playersState, winner) = rawState.as_tuple()

            # extract from playerState our current Player info as "me"
            if playersState[0].enemy:
                me, enemy = playersState[1], playersState[0]
            else:
                me, enemy = playersState[0], playersState[1]

            # store all status in separeted variable
            move_speed = me.moveSpeed   # the player move speed
            bomb = me.bomb              # the number of bomb the player has in his inventory
            bomb_range = me.bombRange   # the explosion range of a player bomb
            (x, z) = me.x, me.z         # the player position in the map
            reward = 0
            done = False

            # transform all data into a one line array (like an classical input layer)
            map_array = np.array(map).flatten()
            status_array = np.array([move_speed, bomb, bomb_range, x, z])
            state_array = np.concatenate(map_array, status_array)

            # convert the state_array into a tensor to use it with pytorch
            tensorState = torch.tensor(state_array, dtype=torch.float32)

            # deduct if the game is done and the current reward of the agent
            if me.dead:
                done = True
                # reward = -1
            if winner:
                done = True
                if winner == self.me.player_num:
                    reward = 1

            return (tensorState, reward, done)
        
        # function that return the Agent action
        # It is the function that will be called by the environment
        def get_action(self, state: State) -> t_action:
            action = 0  # by default do nothing
            # preprocess state data
            (p_state, reward, done) = preprocessState(state)

            # if the game isn't at a beginning state
            # remember and replay last state
            if self.last_p_state:
                self.remember(self.last_p_state, self.last_action, reward, p_state, done)
                self.replay()

            # if the game is done set replay variable to None
            # to ensure that no state will be remembered
            if done:
                self.last_p_state = None
                self.last_action = None
            else:
                # take action from neuralNetwork
                action = self.choose_action(p_state)

                # set current variable for next replay
                self.last_p_state = p_state
                self.last_action = action

            return defines.action_space[action]