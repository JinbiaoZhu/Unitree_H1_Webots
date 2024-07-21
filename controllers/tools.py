"""
Some simple tool functions.
"""
import random
import time

import numpy as np
import torch


def init_weight(net: torch.nn):
    """
    Initialize the net.
    """
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 0.1)
            m.bias.data.zero_()
    return net


def set_seed(env, num=0):
    """
    Set the seed.
    """
    random.seed(num)
    np.random.seed(num)
    env.seed(num)
    torch.manual_seed(num)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def time_string():
    return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))


def print_dict(filetype, d: dict):
    print(f"This file is in {filetype}.")
    for k, v in d.items():
        print(f"{k}\t\t{v}")
    print("\n")


def discounted_rewards(rewards: torch.Tensor, gamma):
    """

    :param gamma:
    :param rewards:
    :return:
    """
    d_rewards = torch.zeros_like(rewards)
    running_reward = 0
    for i in reversed(range(len(rewards))):
        running_reward = running_reward * gamma + rewards[i]
        d_rewards[i] = running_reward

    return d_rewards


def render_interval(i_episode: int, interval: int, env):
    if interval == 0:
        return None
    if i_episode % interval == 0:
        env.render()
