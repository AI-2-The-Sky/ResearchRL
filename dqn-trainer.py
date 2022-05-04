### IMPORTS ###

import gym
import torch
import torch.nn as nn
import gym.envs.toy_text.frozen_lake as frozen_lake

from collections import deque
import math
import random
import csv
import json
import os


### UTILS ###

def random_4x4_lake_map():
    return frozen_lake.generate_random_map(size=4, p=torch.rand(1).item())

def lake_map_to_tensor(lake_map):
    lake_map = ''.join(lake_map)
    mytensor = torch.zeros(len(lake_map), requires_grad=False)
    for i,v in enumerate(lake_map):
        if v == 'H' :
            mytensor[i] = -1
        if v == 'G' :
            mytensor[i] = 1
    return mytensor

def rand_action():
    return math.trunc(torch.rand(1).item() * 4)


### MODEL ###

# model for 4x4 grids
class Model00(nn.Module):
    def __init__(self):
        super(Model00, self).__init__()
        self.dense = nn.Sequential(
             nn.Linear(17, 128),
             nn.ReLU(),
             nn.Linear(128, 32),
             nn.ReLU(),
             nn.Linear(32, 4)
        )

    def forward(self, state, lake_tensor):
        x = torch.cat((torch.as_tensor([state]), lake_tensor))
        x = self.dense(x)
        return x


### TRAINER ###

class MyTrainer():
    def __init__(self, model, is_slippery, discount, epsylon, hundred_runs, replay_memory_max_size, replay_regularity, minibatch_size, loss_fn, output_folder=None):
        self.is_slippery = is_slippery
        self.discount = discount
        self.epsylon = epsylon
        self.hundred_runs = hundred_runs
        self.model = model
        self.loss_fn = loss_fn
        self.replay_regularity = replay_regularity
        self.minibatch_size = minibatch_size
        self.replay_memory_max_size = replay_memory_max_size
        self.folder = output_folder
        if not self.folder :
            self.folder = "tmp"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.state = self.reset_env()
        self.replay_memory = deque(maxlen=replay_memory_max_size)

        self.create_folder()
        self.write_hyperparams()


    def reset_env(self, lake_map=None):
        if not lake_map :
            lake_map = random_4x4_lake_map()

        self.lake_map = lake_map
        self.map_tensor = lake_map_to_tensor(lake_map)
        self.env = gym.make(
            "FrozenLake-v1",
            desc=self.lake_map,
            is_slippery=self.is_slippery
        )
        return self.env.reset()

    def get_epsylon_greedy_action(self, qval):
        action = rand_action()
        if not (qval.max() <= 0 or torch.rand(1).item() < self.epsylon) :
            action = qval.argmax().item()
        return action

    def get_optimal_action(self, qval):
        return qval.argmax().item()

    def bellman_eq(self, action, reward, qval_now, state_next):
        qval_now_updated = qval_now.clone()
        max_qval_next = self.model(state_next, self.map_tensor).max().item()
        qval_now_updated[action] = reward + self.discount * max_qval_next
        return qval_now_updated

    def training(self):
        rewards = []

        rewards_counter = 0
        for t in range(self.hundred_runs * 100):

            state = self.env.reset()
            done = False

            while not done:
                qval = self.model(state, self.map_tensor)
                action = self.get_epsylon_greedy_action(qval)
                next_state, reward, done, info = self.env.step(action)

                self.replay_memory.append(
                    (state, action, reward, next_state)
                )

                state = next_state
                rewards_counter += reward

            if t % 100 == 99 :
                print('Average reward after {} training runs : {}'.format(t+1, rewards_counter/100))
                rewards.append({
                    "total training runs" : t+1,
                    "avg reward over last 100 training runs" : rewards_counter/100
                })
                rewards_counter = 0

            if t % self.replay_regularity == self.replay_regularity - 1:
                self.do_replay_memory(t)

        self.write_training_results(rewards)


    def do_replay_memory(self, t):
        base_lr = 1e-1
        decay_lr = math.e ** (-t/(self.hundred_runs * 100))
        learning_rate = base_lr * decay_lr

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        replays = random.sample(self.replay_memory, self.minibatch_size)
        for (state, action, reward, new_state) in replays:
            qval = self.model(state, self.map_tensor)
            new_qval = self.bellman_eq(action, reward, qval, new_state)
            loss = self.loss_fn(qval, new_qval)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def testing(self):
        rewards = 0
        for t in range(1000):
            #self.reset_env()

            state = self.env.reset()
            done = False
            while not done:
                qval = self.model(state, self.map_tensor)
                action = self.get_optimal_action(qval)
                new_state, reward, done, info = self.env.step(action)
                state = new_state
                rewards += reward

        self.env.close()
        print('Average reward after 1000 runs with optimal strategy : {}'.format(rewards/1000))
        self.write_params()
        self.write_success_rate(rewards/1000)

    def create_folder(self) :
        if not os.path.exists("data/" + self.folder):
            os.makedirs("data/" + self.folder)

    def write_hyperparams(self) :
        try :
            loss_fn_name = self.loss_fn.__class__.__name__
        except :
            try :
                loss_fn_name = self.loss_fn.__name__
            except :
                loss_fn_name = "/"
        try :
            model_name = self.model.__class__.__name__
        except :
            model_name = "/"
        parameters = { "model": model_name, "loss_fn": loss_fn_name, "is_slippery": self.is_slippery, "discount": self.discount, "epsylon": self.epsylon, "hundred_runs": self.hundred_runs, "replay_memory_max_size": self.replay_memory_max_size, "replay_regularity": self.replay_regularity, "minibatch_size": self.minibatch_size }
        json_object = json.dumps(parameters, indent = 4)
        with open("data/" + self.folder + "/hyperparameters.json", "w") as outfile:
            outfile.write(json_object)

    def write_training_results(self, rewards) :
        with open("data/" + self.folder + '/training.csv', 'w') as csvfile:
            columns = ['total training runs', 'avg reward over last 100 training runs']
            writer = csv.DictWriter(csvfile, fieldnames = columns)
            writer.writeheader()
            writer.writerows(rewards)

    def write_params(self) :
        torch.save(self.model.state_dict(), "data/" + self.folder + "/parameters.torch")

    def write_success_rate(self, rate) :
        with open("data/" + self.folder + "/success_rate.txt", "w") as outfile:
            outfile.write(str(rate))


### MAIN ###

if __name__ == "__main__":

    param_dict = {
        "is_slippery" : True,
        "discount" : 0.9,
        "epsylon" : 0.01,
        "hundred_runs" : 5,
        "replay_memory_max_size" : 2000,
        "replay_regularity" : 20,
        "model" : Model00(),
        "loss_fn" : nn.MSELoss(),
        "minibatch_size" : 32
    }

    mt = MyTrainer(**param_dict)
    mt.training()
    mt.testing()