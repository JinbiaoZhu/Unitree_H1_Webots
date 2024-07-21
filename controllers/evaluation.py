"""
Some classes for evaluating reinforcement learning.
"""
import os

import matplotlib.pyplot as plt

from tools import moving_average, time_string


class SimpleEvaluate:
    def __init__(self,
                 save_path: str, algo_name: str, env_name: str, smooth_size=9,
                 requires_smooth=True, requires_loss=False, save_results=False,
                 ):
        if save_results:
            if save_path is None:
                raise ValueError("Must provide a save path.")

        self.save_results = save_results
        self.save_path = save_path

        self.algo_name = algo_name
        self.env_name = env_name

        self.smooth_size = smooth_size

        self.requires_smooth = requires_smooth
        self.requires_loss = requires_loss

        self.episode_return = 0

        self.return_list = []
        if self.requires_loss:
            self.return_loss = []

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def episode_return_is_zero(self):
        self.episode_return = 0

    def episode_return_record(self):
        self.return_list.append(self.episode_return)

    def add_single_step_reward(self, reward: float):
        self.episode_return += reward

    def add_single_update_loss(self, loss):
        if self.requires_loss is False:
            pass
        else:
            self.return_loss.append(loss)

    def plot_performance(self, show=True):
        if self.requires_smooth:
            plot_return_list = moving_average(self.return_list, window_size=self.smooth_size)
        else:
            plot_return_list = self.return_list
        episodes_list = list(range(len(plot_return_list)))

        plt.figure()
        plt.plot(episodes_list, plot_return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title(f'{self.algo_name} on {self.env_name}')
        name = None
        if self.save_results:
            name = self.save_path + f'{self.env_name}-' + time_string() + '.png'
            plt.savefig(name)
        if show:
            plt.show()

        if self.requires_loss:
            plt.figure()
            episodes_list = list(range(len(self.return_loss)))
            plt.plot(episodes_list, self.return_loss)
            plt.legend(["Critic", "Actor"])
            plt.xlabel('Episodes')
            plt.ylabel('Losses')
            plt.title(f'{self.algo_name} on {self.env_name} Loss')
        if self.save_results:
            name += '-loss.png'
            plt.savefig(name)
        if show:
            plt.show()


class EpisodeEvaluate:
    def __init__(self):
        self.episode_total_reward = 0
        self.episode_reward_list = []

    def episode_return_is_zero(self):
        self.episode_total_reward = 0
        self.episode_reward_list.clear()

    def add_single_step_reward(self, reward):
        self.episode_total_reward += reward
        self.episode_reward_list.append(reward)

    def return_total_reward(self):
        return self.episode_total_reward
