"""
Run PPO.
"""
import os

import numpy as np
from tqdm import tqdm

from PPO_algo import PPO
from PPO_config import ppo_config
from evaluation import SimpleEvaluate
from tools import set_seed, print_dict

from UnitreeH1StandingEnv import UnitreeH1StandingV0, UnitreeH1StandingV1

if __name__ == '__main__':

    print_dict(os.path.basename(__file__), ppo_config)

    # env = UnitreeH1StandingV0()
    env = UnitreeH1StandingV1()

    num_episodes = ppo_config["num_episode"]
    num_iteration = ppo_config["num_iteration"]

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    set_seed(env, ppo_config["seed"])

    agent = PPO(state_dim, action_dim,
                hidden_dim=ppo_config["hidden_dim"],
                max_action=max_action,
                lmbda=ppo_config["lmbda"],
                actor_lr=ppo_config["actor_lr"],
                critic_lr=ppo_config["critic_lr"],
                gamma=ppo_config["gamma"],
                device=ppo_config["device"],
                epochs=ppo_config["epoch"],
                epsilon=ppo_config["epsilon"],
                linear=env.metadata["threshold"])

    evaluation = SimpleEvaluate(ppo_config["result_dir"],
                                algo_name=ppo_config["algo"], env_name=ppo_config["env_name"],
                                requires_loss=True, save_results=True)

    for i in range(num_iteration):
        with tqdm(total=int(num_episodes / num_iteration), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / num_iteration)):

                evaluation.episode_return_is_zero()

                state = env.reset()

                while env.render() != -1:
                    action, dist = agent.action(state)
                    next_state, reward, done, info = env.step(action)
                    agent.store(state, action, reward, next_state, done, dist)
                    state = next_state
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
    agent.save(ppo_config["model_dir"])
    env.close()
