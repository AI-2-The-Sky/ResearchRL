from DQN_Agent import BuffedDQNAgent, Hyperparams

from trainer import Trainer
import os

# read wandb api key from env variable
wandb_login_key = os.environ['WANDB_API_KEY']

trainer = Trainer(BuffedDQNAgent, Hyperparams())

trainer.train(plot=False, wandb_login_key=wandb_login_key)
