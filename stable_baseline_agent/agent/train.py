from DQN_Agent import BuffedDQNAgent, Hyperparams

from trainer import Trainer

trainer = Trainer(BuffedDQNAgent, Hyperparams())

trainer.train()