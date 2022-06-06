from os import stat
from turtle import st
from bomberman.agents.BaseAgent		import BaseAgent
from bomberman.states.State			import State, StatePlayer
from bomberman.defines				import t_action
from bomberman						import defines

from random import Random

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Tuple

tiles_watched = {k: i for i, k in enumerate(["B", "E", "W", "C", "r", "b", "s"," "])}


class Hyperparams():
	def __init__(
		self,
		lr : float = 1e-3,
		gamma : float = 0.99,
		optimizer : str = "optim.Adam",
		replay_buffer_size : int = 10000,
		target_update_regularity : int = 100,
		epsilon_decay : float = 1e-3,
		max_epsilon : float = 0.8,
        min_epsilon : float = 0.001,
		# batch_size : int # might want to add this
		reset_concurent_agent_each : int = 10
		training_games_amount : int = 10
		max_game_duration : int = 50
		skip_frames : int = 4 # not clear what this is to me

		# net
		activation_function : str = "nn.LeakyReLu",
		map_conv0_chanels : int = 32,
		map_conv0_kernel_size : int = 5,
		map_conv1_chanels : int = 64,
		map_conv1_kernel_size : int = 3,
		map_conv2_chanels : int = 64,
		map_conv2_kernel_size : int = 3,
		map_avg_pool_kernel_size : int = 3,
		map_linear_chanels : int = 32,
		p_linear0_chanels : int = 28,
		p_linear1_chanels : int = 16,
		act_linear0_chanels : int = 32,
		act_linear1_chanels : int = 16,

		# training techniques
		# TODO : insert in the code @Quentin
		double_dqn : bool = False,
		dueling_learning : bool = False,
		prioritized_experience_replay : bool = False,

	) :
		self.player_amount = player_amount
		self.lr = lr
		self.gamma = gamma
		self.optimizer = optimizer
		self.replay_buffer_size = replay_buffer_size
		self.target_update_regularity = target_update_regularity
		self.epsilon_decay = epsilon_decay
		self.max_epsilon = max_epsilon
		self.min_epsilon = min_epsilon
		self.reset_agent_each = reset_agent_each
		self.training_games_amount = training_games_amount
		self.max_game_duration = max_game_duration
		self.skip_frames = skip_frames
		self.activation_function = activation_function
		self.map_conv0_chanels = map_conv0_chanels
		self.map_conv0_kernel_size = map_conv0_kernel_size
		self.map_conv1_chanels = map_conv1_chanels
		self.map_conv1_kernel_size = map_conv1_kernel_size
		self.map_conv2_chanels = map_conv2_chanels
		self.map_conv2_kernel_size = map_conv2_kernel_size
		self.map_avg_pool_kernel_size = map_avg_pool_kernel_size
		self.map_linear_chanels = map_linear_chanels
		self.p_linear0_chanels = p_linear0_chanels
		self.p_linear1_chanels = p_linear1_chanels
		self.act_linear0_chanels = act_linear0_chanels
		self.act_linear1_chanels = act_linear1_chanels
		self.double_dqn = double_dqn
		self.dueling_learning = dueling_learning
		self.prioritized_experience_replay = prioritized_experience_replay

# TODO : log in mlflow @Simon & @Manu
# something like inspect(class) to get all class params



class NeuralNetwork(nn.Module):
	def __init__(self, tile_types : int, player_info : int, nb_action : int, hp : Hyperparams):
		'''
		tile_types: nb of tiles type (for one hot)
		player_info: number of information per player
		nb_action: action space
		'''
		super().__init__()

		activation_function = exec(hp.activation_function + "()")

		self.map_net = nn.Sequential(
			nn.Conv2d(tile_types, hp.map_conv0_chanels, kernel_size=hp.map_conv0_kernel_size),
			activation_function,
			nn.Conv2d(hp.map_conv0_chanels, hp.map_conv1_chanels, kernel_size=hp.map_conv0_kernel_size),
			activation_function,
			nn.Conv2d(hp.map_conv1_chanels, hp.map_conv2_kernel_size, kernel_size=hp.map_conv2_chanels),
			activation_function,
			nn.AvgPool2d(kernel_size=hp.map_avg_pool_kernel_size),
			nn.Flatten(),
			nn.Linear(hp.map_conv2_chanels, hp.map_linear_chanels),
			activation_function,
		)

		self.p_net = nn.Sequential(
			nn.Linear(player_info + hp.map_linear_chanels, hp.p_linear0_chanels),
			activation_function,
			nn.Linear(hp.p_linear0_chanels, hp.p_linear1_chanels),
			activation_function
		)

		# self.p2_net = nn.Sequential(
		# 	nn.Linear(player_info + hp.map_linear_chanels, hp.p_linear0_chanels),
		# 	activation_function,
		# 	nn.Linear(hp.p_linear0_chanels, hp.p_linear1_chanels),
		# 	activation_function
		# )

		self.action_taker = nn.Sequential(
			nn.Linear(hp.map_linear_chanels + 2*hp.p_linear1_chanels, hp.act_linear0_chanels),
			activation_function,
			nn.Linear(hp.act_linear0_chanels, hp.act_linear1_chanels),
			activation_function,
			nn.Linear(hp.act_linear1_chanels, nb_action),
			nn.Softmax()
		)

	def forward(self, x : Tuple[torch.Tensor]):
		_map, p1, p2 = x

		map_features = self.map_net(_map)
		p1_features = self.p_net(torch.concat([map_features, p1], dim=1))
		# NOTE : same net for both players
		p2_features = self.p_net(torch.concat([map_features, p2], dim=1))
		return self.action_taker(torch.concat([map_features, p1_features, p2_features], dim=1))



class BuffedDQNAgent(BaseAgent):
	'''
	This is agent have these upgrades comparing to vanilla DQN :
		- Double DQN
		- Dueling Learning
		- Prioritized Experience Replay
	Some code is taken from https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/deepq/dqn.py
	'''

	def __init__(self, player_num : int, hp : Hyperparams) -> None:
		super().__init__(hp.player_num)

		self.device = "cuda" if torch.cuda.is_available() else "cpu"

		self.action_space = len(defines.action_space)

		self.brain = NeuralNetwork(len(tiles_watched), 5, self.action_space, hp).to(self.device)

		self.target_brain = NeuralNetwork(len(tiles_watched), 5, self.action_space, hp).to(self.device)
		self.target_brain.load_state_dict(self.brain.state_dict())
		self.target_brain.eval()

		self.optimizer = exec(hp.optimizer + "(self.brain.parameters(), lr=hp.lr)")

		self.hp = hp

	def _raw_state_to_input(self, state : State):
		'''
		Take the current state and process the input that will be sent to the brain
		'''
		_map, players_state, _ = state.as_tuple()

		# First, let's process the map
		processed_map = torch.zeros(11, 11, 8, dtype=torch.float32).to(self.device)
		for y, row in enumerate(_map):
			for x, val in enumerate(row):
				current = [0] * len(tiles_watched)
				current[tiles_watched[val]] = 1
				processed_map[y][x] = torch.tensor(current)
		processed_map = torch.unsqueeze(processed_map.permute(2, 0, 1), dim=0).to(self.device)

		# Then, split players in agent and opponent
		if len(players_state) == 2:
			agent, opponent = players_state[::-1] if players_state[0].enemy else players_state
		elif len(players_state) == 0:
			agent, opponent = StatePlayer(), StatePlayer()
		elif not players_state[0].enemy:
			agent = players_state[0]
			opponent = StatePlayer()
		else:
			opponent = players_state[0]
			agent = StatePlayer()

		agent = torch.tensor([agent.moveSpeed, agent.bombs, agent.bombRange, agent.x, agent.z]).reshape(1, -1).to(self.device)
		opponent = torch.tensor([opponent.moveSpeed, opponent.bombs, opponent.bombRange, opponent.x, opponent.z]).reshape(1,-1).to(self.device)

		return processed_map, agent, opponent

	def update_model(self, data_point):
		loss = self._compute_loss(data_point)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss.item()

	def _compute_loss(self, data_point):
		state = self._raw_state_to_input(data_point["obs"])
		next_state = self._raw_state_to_input(data_point["next_obs"])
		action = torch.LongTensor(np.array(data_point["action"]).reshape(-1, 1)).to(self.device)
		reward = torch.FloatTensor(np.array(data_point["reward"]).reshape(-1, 1)).to(self.device)
		done = torch.FloatTensor(np.array(data_point["done"]).reshape(-1, 1)).to(self.device)

		curr_q_value = self.brain(state).gather(1, action)
		next_q_value = self.target_brain(next_state).max(dim=1, keepdim=True)[0].detach()
		mask = 1 - done
		target = (reward + self.gamma * next_q_value * mask)

		loss = F.smooth_l1_loss(curr_q_value, target)

		return loss

	def _targe_hard_update(self):
		self.target_brain.load_state_dict(self.brain.state_dict())

	def get_action(self, state: State, epsilon=0) -> t_action:
		if epsilon > np.random.random():
			return np.random.choice(self.action_space)

		x = self._raw_state_to_input(state)
		res = torch.argmax(self.brain(x))
		return res.cpu().detach().numpy().tolist()
