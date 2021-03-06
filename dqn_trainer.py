
# ██╗     ███████╗ ██████╗  █████╗  ██████╗██╗   ██╗
# ██║     ██╔════╝██╔════╝ ██╔══██╗██╔════╝╚██╗ ██╔╝
# ██║     █████╗  ██║  ███╗███████║██║      ╚████╔╝
# ██║     ██╔══╝  ██║   ██║██╔══██║██║       ╚██╔╝
# ███████╗███████╗╚██████╔╝██║  ██║╚██████╗   ██║
# ╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝   ╚═╝

#  ██████╗ ██████╗ ██████╗ ███████╗
# ██╔════╝██╔═══██╗██╔══██╗██╔════╝
# ██║     ██║   ██║██║  ██║█████╗
# ██║     ██║   ██║██║  ██║██╔══╝
# ╚██████╗╚██████╔╝██████╔╝███████╗
#  ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝


### IMPORTS ###

import gym
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from collections import deque, OrderedDict
import math
import random
import csv
import json
import os
import copy


### UTILS ###

default_4x4_lake_map = [ "SFFF", "FHFH", "FFFH", "HFFG" ]

def random_4x4_lake_map(fixed_start_and_goal):
    # this is extreamly slow
    # return gym.envs.toy_text.frozen_lake.generate_random_map(size=4, p=r)

    my_map = ['F'] * 16

    rand = torch.normal(torch.tensor([5.0]), torch.tensor([2])).item()
    if rand < 0 : rand = 0
    hole_amount = round(rand)
    for i in torch.rand(hole_amount).tolist() :
        my_map[math.trunc(i*16)] = 'H'

    start, goal = 0, 0
    if fixed_start_and_goal :
        start, goal = 0, 15
    while start == goal :
        se = torch.rand(2) * 16
        start = math.trunc(se[0].item())
        goal = math.trunc(se[1].item())
    my_map[start] = 'S'
    my_map[goal] = 'G'

    return np.reshape(my_map, (4,4)).tolist()


def lake_map_to_tensor(lake_map):
    lake_map = ''.join(np.asarray(lake_map).flatten().tolist())
    mytensor = torch.zeros(len(lake_map), requires_grad=False)
    for i,v in enumerate(lake_map):
        if v == 'H' :
            mytensor[i] = -1
        if v == 'G' :
            mytensor[i] = 1
    return mytensor


def rand_action():
    return math.trunc(torch.rand(1).item() * 4)

def create_folder(folder) :
    if not os.path.exists(folder):
        os.makedirs(folder)

### TRAINER ###

# QNet contien le réseau de neurones aproximant la q-function
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.my_nn = nn.Sequential(
             nn.Linear(17, 128),
             nn.ReLU(),
             nn.Linear(128, 32),
             nn.ReLU(),
             nn.Linear(32, 4)
        )

    def forward(self, state):
        return self.my_nn(state)

# ProcessedState sert à encapsuler les différentes parties du state
# ici ce sont la position du personage ainsi que la carte sur laquelle il joue
class ProcessedState():
    def __init__(self, raw_state, lake_map):
        self.raw_state = raw_state
        self.lake_map = lake_map
        self.map_tensor = lake_map_to_tensor(self.lake_map)

    def update(self, raw_state):
        self.raw_state = raw_state

    def tensor(self):
        return torch.cat((torch.as_tensor([self.raw_state]), self.map_tensor))


# MemoryReplay conserve les states dans lesquels s'est retrouvé le modèle lors
# de son entrainement pour les restituer sous forme de batches
class MemoryReplay():
    def __init__(self, replay_memory_max_size:int, minibatch_size:int):
        self.replay_memory_max_size = replay_memory_max_size
        self.replay_memory = deque(maxlen=replay_memory_max_size)
        self.minibatch_size = minibatch_size

    def store(self, state:ProcessedState, action:int, reward:float, next_state:ProcessedState, done:bool):
        self.replay_memory.append({
            "s0":state,
            "a":action,
            "r":reward,
            "s1":next_state,
            "d":done
        })

    def sample(self):
        sample = random.sample( self.replay_memory, self.minibatch_size)
        random.shuffle(sample)
        return sample

    def sample_as_tensor(self):
        sample = self.sample()
        keys = ['s0', 'a', 'r', 's1', 'd']
        sample_dict = {key : [] for key in keys}
        for s in sample :
            sample_dict['s0'].append(s['s0'].tensor())
            sample_dict['s1'].append(s['s1'].tensor())
            sample_dict['a'].append(s['a'])
            sample_dict['r'].append(s['r'])
            sample_dict['d'].append(s['d'])
        sample_tensor_dict = {
            's0': torch.stack(sample_dict['s0']),
            's1': torch.stack(sample_dict['s1']),
            'a': torch.tensor(sample_dict['a']),
            'r': torch.tensor(sample_dict['r']),
            'd': torch.tensor(sample_dict['d']),
        }
        return sample_tensor_dict

    def is_large_enough(self):
        return (len(self.replay_memory) > self.minibatch_size)

    def save(self, folder:str):
        mem_params = {
            "replay_memory_max_size": self.replay_memory_max_size,
            "minibatch_size": self.minibatch_size,
        }
        with open(folder + "/memory_replay.json", "w") as outfile:
            outfile.write(json.dumps(mem_params, indent = 4))


# Guilhem regarde ici ;)
# la classe Environment sert à encapsuler l'environment et à stendardiser ses in et outputs
class Environment():
    def __init__(self, is_slippery:bool, randomize_lake_map:bool, fixed_start_and_goal:bool, custom_map=None):
        self.randomize_lake_map = randomize_lake_map
        self.fixed_start_and_goal = fixed_start_and_goal
        self.is_slippery = is_slippery

        self.lake_map = default_4x4_lake_map if not custom_map else custom_map
        self.env = None
        self.state = None
        self.reset()

    def reset(self):
        if self.randomize_lake_map :
            self.lake_map = random_4x4_lake_map(self.fixed_start_and_goal)
            if self.env :
                self.close()
            self.env = gym.make(
                "FrozenLake-v1",
                desc=self.lake_map,
                is_slippery=self.is_slippery
            )
            self.state = ProcessedState(self.env.reset(), self.lake_map)
        elif not self.state or not self.env :
            self.env = gym.make(
                "FrozenLake-v1",
                desc=self.lake_map,
                is_slippery=self.is_slippery
            )
            self.state = ProcessedState(self.env.reset(), self.lake_map)
        else :
            self.state.update(self.env.reset())

    def step(self, action:int):
        new_raw_state, reward, done, _ = self.env.step(action)
        old_state = copy.deepcopy(self.state)
        self.state.update(new_raw_state)
        return old_state, self.state, reward, done

    def close(self):
        self.env.close()
        self.state = None

    def save(self, folder:str):
        env_params = {
            "is_slippery": self.is_slippery,
            "randomize_lake_map": self.randomize_lake_map,
            "fixed_start_and_goal": self.fixed_start_and_goal,
            "map": self.lake_map
        }
        with open(folder + "/environment.json", "w") as outfile:
            outfile.write(json.dumps(env_params, indent = 4))


class Agent():
    def __init__(self, model:QNet, discount:float, epsilon:float, epsilon_decay_rate:float, epsilon_min:float, lr:float, lr_decay_rate:float, weight_decay:float, memory_replay:MemoryReplay):
        self.model = model
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.memory_replay = memory_replay

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model_target = copy.deepcopy(model)

        # unexposed hyperparam
        self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.lr_decay_rate)
        self.loss_fn = nn.MSELoss()


    def choose_optimal_action(self, state:ProcessedState):
        return self.model(state.tensor()).argmax().item()

    def choose_epsilon_greedy_action(self, state:ProcessedState):
        qval = self.model(state.tensor())
        action = rand_action()
        if not (qval.max() <= 0 or torch.rand(1).item() < self.epsilon) :
            action = qval.argmax().item()
        return action


    def model_backprop(self):
        if self.memory_replay.is_large_enough():
            print(len(self.memory_replay.replay_memory))
            samples = self.memory_replay.sample_as_tensor()
            print(samples['s0'].size())
            pred_qvals = self.model(samples['s0'])

            next_best_actions = self.model(samples['s1']).argmax(dim=-1)
            next_qvals = self.model_target(samples['s1'])

            updated_qvals = pred_qvals.clone()
            for i, (next_best_action, next_qval, action, reward, done) in enumerate(zip(next_best_actions, next_qvals, samples['a'], samples['r'], samples['d'])) :
                if done :
                    updated_qvals[i][action] = reward
                else :
                    updated_qvals[i][action] = reward + self.discount * next_qval[next_best_action]

            loss = self._compute_loss(pred_qvals, updated_qvals)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self._epsilon_decay_step()
            self._target_model_update()

    def _epsilon_decay_step(self):
        if self.epsilon - self.epsilon_decay_rate - self.epsilon_min <= 0 :
            self.epsilon -= self.epsilon_decay_rate

    def _compute_loss(self, pred_y, y):
        return self.loss_fn(pred_y, y)

    def _target_model_update(self):
        self.model_target = copy.deepcopy(self.model)

    def save(self, folder:str):
        self._save_params(folder)
        self._save_hyperparams(folder)

    def _save_params(self, folder:str) :
        torch.save(self.model.state_dict(), folder + "/parameters.torch")

    def _save_hyperparams(self, folder:str) :
        try :
            model_name = self.model.__class__.__name__
        except :
            model_name = "?"
        try :
            optimizer_name = self.optimizer.__class__.__name__
        except :
            optimizer_name = "?"
        try :
            lr_scheduler_name = self.lr_scheduler.__class__.__name__
        except :
            lr_scheduler_name = "?"
        try :
            loss_fn_name = self.loss_fn.__class__.__name__
        except :
            loss_fn_name = "?"
        hyperparameters = {
            "model": model_name,
            "lr_scheduler": lr_scheduler_name,
            "optimizer": optimizer_name,
            "loss_fn": loss_fn_name,
            "discount": self.discount,
            "epsilon": self.epsilon,
            "epsilon_decay_rate": self.epsilon_decay_rate,
            "epsilon_min": self.epsilon_min,
            "weight_decay": self.weight_decay,
            "learning_rate": self.lr,
            "learning_rate_decay": self.lr_decay_rate,
            "device": self.device,
        }
        with open(folder + "/hyperparameters.json", "w") as outfile:
            outfile.write(json.dumps(hyperparameters, indent = 4))


class Trainer() :
    def __init__(self, environment:Environment, agent:Agent):
        self.agent = agent
        self.env = environment
        self.memory = agent.memory_replay
        self.training_data = {}
        self.testing_data = []

    # la fonction train sert à entrainer l'agent.
    # 1. en générant des exemples et en les enregistrant dans le memory_replay
    # 2. en appelant agent.model_backprop() pour qu'il apprenne sur base de ces exemples
    def train(self, timesteps:int, backprop_regularity:int):
        training_id = len(self.training_data)
        self.training_data[training_id] = []
        rewards_counter = 0
        done_counter = 0
        for t in range(timesteps) :
            action = self.agent.choose_epsilon_greedy_action(self.env.state)
            old_state, new_state, reward, done = self.env.step(action)
            self.memory.store(old_state, action, reward, new_state, done)
            rewards_counter += reward
            done_counter += 1 if done else 0
            if (done_counter + 1) % backprop_regularity == 0 :
                print('Average reward after {} training runs : {}'.format(done_counter + 1, rewards_counter / backprop_regularity))
                rewards_counter = 0
                self.training_data[training_id].append((done_counter, rewards_counter / backprop_regularity))
                self.agent.model_backprop()
            if done:
                self.env.reset()


    # la fonction test sert à tester l'agent dans ses meilleurs conditions
    # en fesant jouer l'agent et en gardant trace de ses performances
    def test(self, runs:int=1000):
        rewards_counter = 0
        for t in range(runs):
            done = False
            while not done:
                action = self.agent.choose_optimal_action(self.env.state)
                old_state, new_state, reward, done = self.env.step(action)
                rewards_counter += reward
            self.env.reset()
        print('Average reward after {} testing runs : {}'.format(runs, rewards_counter / runs))
        self.testing_data.append((runs, rewards_counter / runs))


    #def step(self, action:int):
    #    next_state, reward, done, info = self.env.step(action)
    # def plot(self):
    #     agent.choose_epsilon_greedy_action()

    def save(self, folder:str):
        self._save_training_data(folder)
        self._save_testing_data(folder)

    def _save_training_data(self, folder:str):
        for i, td in self.training_data.items() :
            print(type(td[0]))
            with open(folder + '/training_' + str(i) + '.csv', 'w') as csvfile:
                columns = ['total training runs', 'avg reward over last 100 training runs']
                writer = csv.DictWriter(csvfile, fieldnames = columns)
                writer.writeheader()
                writer.writerow(td)


    def _save_testing_data(self, folder:str):
        with open(folder + "/testing.csv", "w") as csvfile:
            columns = ['testing runs amount', 'avg reward over testing runs']
            writer = csv.DictWriter(csvfile, fieldnames = columns)
            writer.writeheader()
            writer.writerows(self.testing_data)



### MAIN ###

if __name__ == "__main__":
    model = QNet()
    mem_params = {
        "replay_memory_max_size" : 2**11,
        "minibatch_size" : 64,
    }
    memory = MemoryReplay(**mem_params)
    ag_params = {
        "model" : model,
        "discount" : 0.9,
        "epsilon" : 0.1,
        "epsilon_decay_rate" : 0.001,
        "epsilon_min" : 0.001,
        "lr" : 0.01,
        "lr_decay_rate" : 1e-5,
        "weight_decay" : 1e-8,
        "memory_replay" : memory,
    }
    agent = Agent(**ag_params)
    env_params = {
        "is_slippery" : True,
        "randomize_lake_map" : False,
        "fixed_start_and_goal" : True,
    }
    environment = Environment(**env_params)
    trainer = Trainer(environment, agent)

    trainer.train(int(1e6), 100) ##initialise
    trainer.test()

    folder = "tmp_data"
    create_folder(folder)
    memory.save(folder)
    agent.save(folder)
    environment.save(folder)
    trainer.save(folder)
