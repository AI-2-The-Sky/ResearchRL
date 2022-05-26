from os import stat
from bomberman.agents.BaseAgent		import BaseAgent
from bomberman.states.State			import State
from bomberman.defines				import t_action
from bomberman						import defines

from random import Random

import torch
import torch.nn as nn
import numpy as np

from typing import Tuple

tiles_watched = {k: i for i, k in enumerate(["B", "E", "W", "C", "r", "b", "s"," "])}

class NeuralNetwork(nn.Module):
	def __init__(self, tile_types : int, player_info : int, nb_action : int):
		'''
		tile_types: nb of tiles type (for one hot)
		player_info: number of information per player
		nb_action: action space
		'''
		super().__init__()

		
		self.map_net = nn.Sequential(
			nn.Conv2d(tile_types, 32, kernel_size=5),
			nn.LeakyReLU(),
			nn.Conv2d(32, 64, kernel_size=3),
			nn.LeakyReLU(),
			nn.Conv2d(64, 64, kernel_size=3),
			nn.LeakyReLU(),
			nn.AvgPool2d(kernel_size=3),
			nn.Flatten(),
			nn.Linear(64, 32),
			nn.LeakyReLU(),
		)

		self.p1_net = nn.Sequential(
			nn.Linear(player_info + 32, 28),
			nn.LeakyReLU(),
			nn.Linear(28, 16),
			nn.LeakyReLU()
		)

		self.p2_net = nn.Sequential(
			nn.Linear(player_info + 32, 28),
			nn.LeakyReLU(),
			nn.Linear(28, 16),
			nn.LeakyReLU()
		)


		self.action_taker = nn.Sequential(
			nn.Linear(64, 32),
			nn.LeakyReLU(),
			nn.Linear(32, 16),
			nn.LeakyReLU(),
			nn.Linear(16, nb_action),
			nn.Softmax()
		)

	def forward(self, x : Tuple[torch.Tensor]):
		_map, p1, p2 = x

		map_features = self.map_net(_map)
		p1_features = self.p1_net(torch.concat([map_features, p1], dim=1))
		p2_features = self.p2_net(torch.concat([map_features, p2], dim=1))
		return self.action_taker(torch.concat([map_features, p1_features, p2_features], dim=1))

class BuffedDQNAgent(BaseAgent):
	'''
	This is agent have these upgrades comparing to vanilla DQN :
		- Double DQN
		- Dueling Learning
		- Prioritized Experience Replay
	Some code is taken from https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/deepq/dqn.py
	'''
	
	def __init__(self, player_num: int) -> None:
		super().__init__(player_num)

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.brain = NeuralNetwork(len(tiles_watched), 5, len(defines.action_space)).to(self.device)


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
		agent, opponent = players_state[::-1] if players_state[0].enemy else players_state
		agent = torch.tensor([agent.moveSpeed, agent.bombs, agent.bombRange, agent.x, agent.z]).reshape(1, -1).to(self.device)
		opponent = torch.tensor([opponent.moveSpeed, opponent.bombs, opponent.bombRange, opponent.x, opponent.z]).reshape(1,-1).to(self.device)

		return processed_map, agent, opponent

	def get_action(self, state: State, train=False) -> t_action:
		x = self._raw_state_to_input(state)
		res = torch.argmax(self.brain(x))
		if not train:
			return res.cpu().detach().numpy().tolist()
		else:
			return res 