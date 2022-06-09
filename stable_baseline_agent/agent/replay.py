from typing import Dict, List
from bomberman.states.State import State
import numpy as np


class ReplayBuffer():
    """A simple replay buffer without batch."""
    def __init__(self, size: int):
        self.buffer = []

        self.max_size = size

    def store(
        self,
        data_point: Dict[str, State or List[int] or float or bool],
    ):
        self.buffer.append(data_point)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(-1)

    def sample_batch(self) -> Dict[str, State or List[int] or float or bool]:
        return self.buffer[np.random.choice(len(self))]

    def __len__(self) -> int:
        return len(self.buffer)