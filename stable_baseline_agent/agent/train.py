from DQN_Agent import BuffedDQNAgent, Hyperparams

from trainer import Trainer

# TODO : give your wandb login key
wandb_login_key = None

trainer = Trainer(BuffedDQNAgent, Hyperparams())

trainer.train(plot=False, wandb_login_key=wandb_login_key)
