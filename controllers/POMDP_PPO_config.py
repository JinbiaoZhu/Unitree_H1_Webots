"""
A dict recording the important hyperparameters.
"""
from base_declaration import *

pomdp_ppo_config = {
    "algo": "POMDP_PPO",
    "actor_lr": 0.00001,
    "critic_lr": 0.0001,
    "num_episode": 5000,
    "num_iteration": 10,
    "gamma": 0.9,
    "lmbda": 0.9,
    "hidden_dim": 128,
    "recurrent_layers": 2,
    "epoch": 1,
    "epsilon": 0.2,
    "is_trainable_std_dev": True,
    "init_log_std_dev": 0.0,
    "seed": 0,
    "env_name": 'UnitreeH1_Standing',
    "model_dir": './ckpts/pomdp_ppo/',
    "result_dir": './results/pomdp_ppo/',
    "device": DEVICE1,
    "dtype": DTYPE,
    "eps": EPS
}
