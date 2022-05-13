# run this cmd for fast iteration (it runs the program on file save) :
# nodemon --exec "python3 dqn_trainer.py" -w dqn_trainer.py


### IMPORTS ###

import gym
import torch
import torch.nn as nn
import numpy as np

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
        return torch.cat((torch.as_tensor([self.raw_state]), self.lake_tensor))


# MemoryReplay conserve les states dans lesquels s'est retrouvé le modèle lors
# de son entrainement pour les restituer sous forme de batches
class MemoryReplay():
    def __init__(self, replay_memory_max_size:int, minibatch_size:int):
        self.replay_memory_max_size = replay_memory_max_size
        self.replay_memory = deque(maxlen=replay_memory_max_size)

    def store(self, state:ProcessedState, action:int, reward:float, next_state:ProcessedState, done:bool):
        self.replay_memory.append(
            (state, action, reward, next_state, done)
        )

    def sample(self):
        return random.shuffle(random.sample( self.replay_memory, self.minibatch_size))

    def sample_as_tensor(self):
        sample = self.sample(self.minibatch_size)
        ls0, la0, lr, ls1, ldone = [], [], [], [], []
        df = pd.DataFrame(sample, columns=['states', 'actions', 'rewards', 'next_states', 'done'])
        out = {}
        for column in df.columns :
            out[column] = torch.tensor(df[column].values)
        return out


# Guilhem regarde ici ;)
# la classe Environment sert à encapsuler l'enviroment et à stendardiser ses in et outputs
class Environment():
    def __init__(self, is_slippery:bool, randomize_lake_map:bool, fixed_start_and_goal:bool, custom_map=None):
        self.randomize_lake_map = randomize_lake_map
        self.fixed_start_and_goal = fixed_start_and_goal
        self.is_slippery = is_slippery

        self.lake_map = None
        self.env = None
        self.state = None
        self.reset()

    def reset(self):
        if self.randomize_lake_map :
            self.lake_map = random_4x4_lake_map(self.fixed_start_and_goal)
            if self.env :
                self.env.close()
            self.env = gym.make(
                "FrozenLake-v1",
                desc=self.lake_map,
                is_slippery=self.is_slippery
            )
            self.state = ProcessedState(self.env.reset(), self.lake_map)
        elif not self.lake_map :
            if not self.custom_map :
                self.lake_map = default_4x4_lake_map
            else :
                self.lake_map = self.custom_map
            self.env = gym.make(
                "FrozenLake-v1",
                desc=self.lake_map,
                is_slippery=self.is_slippery
            )
            self.state = ProcessedState(self.env.reset(), self.lake_map)

    def step(self, action:int):
        new_raw_state, reward, done, _ = self.enf.step(action)
        old_state = self.state
        new_state = self.state.new(new_raw_state)
        self.state = new_state
        return old_state, new_state, reward, done


class Agent():
    def __init__(self, model:QNet, discount:float, epsilon:float, epsilon_decay_rate:float, epsilon_min:float, lr:float, lr_decay_rate:float, weight_decay:float, memory_replay:MemoryReplay):
        self.model = model
        self.model_target = copy.deepcopy(model)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay = weight_decay
        self.memory_replay = memory_replay

        # unexposed hyperparam
        self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.lr_decay_rate)
        self.loss_fn = nn.MSELosse()


    def choose_optimal_action(self, state:ProcessedState):
        return self.model(state.tensor()).argmax().item()

    def choose_epsilon_greedy_action(self, state:ProcessedState):
        qval = self.model(state.tensor())
        action = rand_action()
        if not (qval.max() <= 0 or torch.rand(1).item() < self.epsilon) :
            action = qval.argmax().item()
        return action

    def epsilon_decay_step(self):
        if self.epsilon - self.epsilon_decay_rate - self.epsilon_min <= 0 :
            self.epsilon -= self.epsilon_decay_rate

    def model_backprop(self):
        samples = self.memory_replay.sample_as_tensor()
        pred_qvals = self.model(samples['states'])

        next_best_actions = self.model(samples['next_states']).argmax(dim=-1)
        next_qvals = self.model_target(samples['next_states'])

        updated_qvals = pred_qvals.clone()
        for i, (next_best_action, next_qval, reward, done) in enumerate(zip(next_best_actions, next_qvals, samples['reward'], samples['done'])) :
            if done :
                updated_qvals[i][action] = reward
            else :
                updated_qvals[i][action] = reward + self.discount * next_qval[next_best_action]

        loss = self._compute_loss(pred_qvals, updated_qvals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._target_update()

    def _compute_loss(self, pred_y, y):
        return self.loss_fn(pred_y, y)

    def _target_update(self):
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
            "minibatch_size": self.minibatch_size,
            "weight_decay": self.weight_decay,
            "learning_rate": self.lr,
            "learning_rate_decay": self.lr_decay_rate,
            "device": self.device,
            "replay_memory_max_size": self.replay_memory.replay_memory_max_size
        }
        with open(folder + "/hyperparameters.json", "w") as outfile:
            outfile.write(json.dumps(hyperparameters, indent = 4))


class Trainer() :
    def __init__(self, environment:Environment, agent:Agent):
        self.agent = agent
        self.env = environment
        self.memory = agent.memory_replay

    # la fonction train sert à entrainer l'agent.
    # 1. en générant des exemples et en les enregistrant dans le memory_replay
    # 2. en appelant agent.model_backprop() pour qu'il apprenne sur base de ces exemples
    def train(self, timesteps:int, replay_every:int):
        rewards_counter = 0
        done_counter = 0
        for t in range(timesteps)
            action = agent.choose_epsilon_greedy_action(env.state)
            old_state, new_state, reward, done = self.env.step(action)
            memory.store(old_state, action, reward, new_state)
            rewards_counter += reward
            done_counter += done
            if not (done_counter % 1000):
                print('Average reward after {} training runs : {}%'.format(done_counter, rewards_counter / 1000))
                rewards_counter = 0
            if not (t % replay_every):
                agent.model_backprop()
            if done:
                self.env.reset()


    # la fonction test sert à tester l'agent dans ses meilleurs conditions
    # en fesant jouer l'agent et en gardant trace de ses performances
    def test(self, runs:int=1000):
        reward = 0
        for t in range(runs):
            done = false
            while not done:
                action = agent.choose_optimal_action(env.state)
                old_state, new_state, reward, done = self.env.step(action)
                rewards_counter += reward
            self.env.reset()
        print('Average reward after {} training runs : {}%'.format(runs, rewards_counter / runs))


    #def step(self, action:int):
    #    next_state, reward, done, info = self.env.step(action)
    # def plot(self):
    #     agent.choose_epsilon_greedy_action()

    def _create_save_folder(self) :
        if not os.path.exists("dqn-trainer-data/" + self.folder):
            os.makedirs("dqn-trainer-data/" + self.folder)

    def _save_training_parameters(self, rewards) :
        with open("dqn-trainer-data/" + self.folder + '/training.csv', 'w') as csvfile:
            columns = ['total training runs', 'avg reward over last 100 training runs']
            writer = csv.DictWriter(csvfile, fieldnames = columns)
            writer.writeheader()
            writer.writerows(rewards)

    def _save_success_rate(self, rate) :
        with open("dqn-trainer-data/" + self.folder + "/success_rate.txt", "w") as outfile:
            outfile.write(str(rate))
        self.env.close()
        self.write_training_results(rewards)


# Old code

class MyTrainer():
    def __init__(self, model, is_slippery, randomize_lake_map, fixed_start_and_goal, discount, epsilon, d_runs, replay_memory_max_size, replay_regularity, minibatch_size, weight_decay, learning_rate, learning_rate_decay, output_folder=None, custom_map=None):
        # the unexposed hyperparameters are :
        # optimizer, loss_fn and lr_scheduler functions
        # might have a few other

        #Agent
        self.model = model
        self.target_model = copy.deepcopy(self.model)
        self.discount = discount
        self.epsilon = epsilon
        self.d_runs = d_runs
        self.replay_regularity = replay_regularity
        self.minibatch_size = minibatch_size
        self.replay_memory_max_size = replay_memory_max_size
        self.replay_memory = deque(maxlen=replay_memory_max_size)

        self.loss_fn = nn.MSELoss()
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.learning_rate_decay)

        self.is_slippery = is_slippery
        self.randomize_lake_map = randomize_lake_map
        self.fixed_start_and_goal = fixed_start_and_goal
        self.custom_map = custom_map
        self.lake_map, self.lake_tensor = None, None
        self.state = self.reset_env()

        self.folder = output_folder
        if not self.folder :
            self.folder = "tmp"
        self.create_folder()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.write_hyperparams()

    # Trainer

    def run(self):
        self.training()
        self.testing()


    def reset_env(self):
        if self.randomize_lake_map :
            self.lake_map = random_4x4_lake_map(self.fixed_start_and_goal)
            self.map_tensor = lake_map_to_tensor(self.lake_map)
            if "self.env" in locals() :
                self.env.close()
            self.env = gym.make(
                "FrozenLake-v1",
                desc=self.lake_map,
                is_slippery=self.is_slippery
            )
        elif not self.lake_map or not self.lake_tensor :
            if not self.custom_map :
                self.lake_map = default_4x4_lake_map
            else :
                self.lake_map = self.custom_map
            self.map_tensor = lake_map_to_tensor(self.lake_map)
            self.env = gym.make(
                "FrozenLake-v1",
                desc=self.lake_map,
                is_slippery=self.is_slippery
            )
        return self.env.reset()

    def training(self): # FIXME
        rewards = []

        rewards_counter = 0
        for t in range(self.hundred_runs * 100):

            state = self.reset_env()

            done = False
            while not done:
                # agent
                qval = self.model(state, self.map_tensor)
                action = self.get_epsilon_greedy_action(qval)
                # trainer
                next_state, reward, done, info = self.env.step(action)

                self.replay_memory.append(
                    (state, action, reward, next_state, done)
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


    def testing(self):
        rewards = 0
        for t in range(1000):
            state = self.reset_env()

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
        if not os.path.exists("dqn-trainer-data/" + self.folder):
            os.makedirs("dqn-trainer-data/" + self.folder)

    def write_hyperparams(self) :
        try :
            model_name = self.model.__class__.__name__
        except :
            model_name = "?"
        parameters = {
            "model": model_name,
            "is_slippery": self.is_slippery,
            "randomize_lake_map": self.randomize_lake_map,
            "fixed_start_and_goal": self.fixed_start_and_goal,
            "custom_map" : self.custom_map,
            "discount": self.discount,
            "epsilon": self.epsilon,
            "hundred_runs": self.hundred_runs,
            "replay_memory_max_size": self.replay_memory_max_size,
            "replay_regularity": self.replay_regularity,
            "minibatch_size": self.minibatch_size,
            "weight_decay": self.weight_decay,
            "learning_rate": self.learning_rate,
            "learning_rate_decay": self.learning_rate_decay,
            "device": self.device
        }
        json_object = json.dumps(parameters, indent = 4)
        with open("dqn-trainer-data/" + self.folder + "/hyperparameters.json", "w") as outfile:
            outfile.write(json_object)


    def write_training_results(self, rewards) :
        with open("dqn-trainer-data/" + self.folder + '/training.csv', 'w') as csvfile:
            columns = ['total training runs', 'avg reward over last 100 training runs']
            writer = csv.DictWriter(csvfile, fieldnames = columns)
            writer.writeheader()
            writer.writerows(rewards)


    def write_params(self) :
        torch.save(self.model.state_dict(), "dqn-trainer-data/" + self.folder + "/parameters.torch")


    def write_success_rate(self, rate) :
        with open("dqn-trainer-data/" + self.folder + "/success_rate.txt", "w") as outfile:
            outfile.write(str(rate))
        self.env.close()
        self.write_training_results(rewards)

    # Agent

    def get_epsilon_greedy_action(self, qval):
        action = rand_action()
        if not (qval.max() <= 0 or torch.rand(1).item() < self.epsilon) :
            action = qval.argmax().item()
        return action


    def get_optimal_action(self, qval):
        return qval.argmax().item()


    def bellman_eq(self, action, reward, qval_now, state_next):
        qval_now_updated = qval
        # ??? some lines missing
        qval_now_updated[action] = reward + self.discount * max_qval_next
        return qval_now_updated
        self.write_training_results(rewards)


    def do_replay_memory(self, t):
        replays = random.sample(
            self.replay_memory,
            min( round( len(self.replay_memory)/5 ), self.minibatch_size )
        )
        for (state, action, reward, new_state, done) in replays:
            qval = self.model(state, self.map_tensor)
            new_qval = reward if done else self.bellman_eq(action, reward, qval, new_state)
            loss = self.loss_fn(qval, new_qval)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if len(replays) == 0 :
        with open("dqn-trainer-data/" + self.folder + "/success_rate.txt", "w") as outfile:
            outfile.write(str(rate))

# states, actions, rewards, next_states, dones = self.sample_memory()

# indices = np.arange(self.batch_size)
# q_pred = self.q_net.forward(states)[indices, actions]
# q_next = self._get_qnext(next_states)

# q_next[dones] = 0.0
# q_target = rewards + self.gamma * q_next

# loss = self.q_net.loss(q_target, q_pred).to(self.q_net.device)
# loss.backward()
# self.q_net.optimizer.step()

### MODEL ###

# model for 4x4 grids
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.my_nn = nn.Sequential(
             nn.Linear(17, 128),
             nn.ReLU(),
             nn.Linear(128, 32),
             nn.ReLU(),
             nn.Linear(32, 4)
        )

    def forward(self, state, lake_tensor):
        x = torch.cat((torch.as_tensor([state]), lake_tensor))
        x = self.my_nn(x)
        return x


### MAIN ###

if __name__ == "__main__":

    param_dict = {
        "is_slippery" : True,
        "randomize_lake_map" : True,
        "fixed_start_and_goal" : True,
        "discount" : 0.9,
        "epsilon" : 0.05,
        "hundred_runs" : 10,
        "replay_memory_max_size" : 8192,
        "replay_regularity" : 20,
        "model" : MyModel(),
        "minibatch_size" : 128,
        "weight_decay": 1e-8,
        "learning_rate": 0.01,
        "learning_rate_decay": 0,
        "output_folder" : None
    }
    MyTrainer(**param_dict).run()
