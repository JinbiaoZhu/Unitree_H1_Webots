"""
Run POMDP_PPO.
"""
import os

import numpy as np
import torch
from tqdm import tqdm

from POMDP_PPO_algo import PomdpPPO
from POMDP_PPO_config import pomdp_ppo_config
from evaluation import SimpleEvaluate
from tools import set_seed, print_dict

from UnitreeH1StandingEnv import UnitreeH1StandingV1

if __name__ == '__main__':

    print_dict(os.path.basename(__file__), pomdp_ppo_config)

    env = UnitreeH1StandingV1()

    num_episodes = pomdp_ppo_config["num_episode"]
    num_iteration = pomdp_ppo_config["num_iteration"]

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    set_seed(env, pomdp_ppo_config["seed"])

    agent = PomdpPPO(state_dim=state_dim,
                     action_dim=action_dim,
                     hidden_dim=pomdp_ppo_config["hidden_dim"],
                     recurrent_layers=pomdp_ppo_config["recurrent_layers"],
                     max_action=max_action,
                     actor_lr=pomdp_ppo_config["actor_lr"],
                     critic_lr=pomdp_ppo_config["critic_lr"],
                     is_trainable_std_dev=pomdp_ppo_config["is_trainable_std_dev"],
                     init_log_std_dev=pomdp_ppo_config["init_log_std_dev"],
                     lmbda=pomdp_ppo_config["lmbda"],
                     gamma=pomdp_ppo_config["gamma"],
                     device=pomdp_ppo_config["device"],
                     dtype=pomdp_ppo_config["dtype"],
                     epochs=pomdp_ppo_config["epoch"],
                     epsilon=pomdp_ppo_config["epsilon"],
                     linear=env.metadata["threshold"])

    evaluation = SimpleEvaluate(pomdp_ppo_config["result_dir"],
                                algo_name=pomdp_ppo_config["algo"], env_name=pomdp_ppo_config["env_name"],
                                requires_loss=True, save_results=True)

    for i in range(num_iteration):
        with tqdm(total=int(num_episodes / num_iteration), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / num_iteration)):

                evaluation.episode_return_is_zero()

                with torch.no_grad():
                    state = env.reset()

                    while env.render() != -1:
                        action, dist = agent.action(state)
                        next_state, reward, done, info = env.step(action)
                        agent.store(state, action, reward, next_state, done, dist)
                        state = np.asarray(next_state)
                        evaluation.add_single_step_reward(reward)

                        if done:
                            break

                evaluation.episode_return_record()

                lv = agent.update()
                evaluation.add_single_update_loss(lv)

                if (i_episode + 1) % num_iteration == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / num_iteration * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(evaluation.return_list[-10:])
                    })
                pbar.update(1)

    evaluation.plot_performance()
    agent.save(pomdp_ppo_config["model_dir"])
    env.close()
