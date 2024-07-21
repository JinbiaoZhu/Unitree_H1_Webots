"""
Replay buffers.
"""
import collections
import copy
from collections import deque
import random
from typing import Dict
import numpy as np
from base_declaration import *


class ReplayBuffer:
    """
    A simple replay buffer.
    """

    def __init__(self, maxlen, batchsize):
        self._maxlen = maxlen
        self._batchsize = batchsize
        self.buffer = collections.deque(maxlen=self._maxlen)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        minibatch = random.sample(self.buffer, self._batchsize)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        batch_dict = {"state": np.array(states),
                      "action": actions,
                      "reward": rewards,
                      "next_state": np.array(next_states),
                      "done": dones}
        return batch_dict

    def size(self):
        return len(self.buffer)


class Episode:
    def __init__(self,
                 requires_distribution=True,
                 requires_goal=False):
        self._requires_distribution = requires_distribution
        self._requires_goal = requires_goal
        self.episode_buffer = collections.deque()

    def add(self, state, action, reward, next_state, done, dist):
        experience = [state, action, reward, next_state, done, dist]
        self.episode_buffer.append(experience)

    def obtain_dict(self):
        if not self._requires_goal:
            states, actions, rewards, next_states, dones, dists = zip(*self.episode_buffer)
            a_dict = {"state": np.asarray(states),
                      "action": np.asarray(actions),
                      "reward": np.asarray(rewards),
                      "next_state": np.asarray(next_states),
                      "done": np.asarray(dones), }
            if self._requires_distribution:
                a_dict["dist"] = dists
        else:
            obs, actions, rewards, next_obs, dones, dists, g, ag = zip(*self.episode_buffer)
            a_dict = {"state": obs,
                      "action": actions,
                      "reward": rewards,
                      "next_state": next_obs,
                      "done": dones,
                      "dist": dists,
                      "goal": g,
                      "achieved_goal": ag}
        return a_dict

    def obtain_episode(self):
        return self.episode_buffer

    def clear(self):
        self.episode_buffer = self.episode_buffer.clear()

    @property
    def requires_distribution(self):
        return self._requires_distribution

    @property
    def requires_goal(self):
        return self._requires_goal


class SequenceReplayBuffer:
    """
    A replay buffer with sequential episodes.
    """

    def __init__(self, maxlen, batchsize, requires_dists=True):
        self._maxlen = maxlen
        self._batchsize = batchsize
        self._requires_dists = requires_dists

        self.buffer = collections.deque(maxlen=self._maxlen)

        self.episode_buffer = Episode(self._requires_dists)

    def add_to_episode(self, state, action, reward, next_state, done, dist):
        self.episode_buffer.add(state, action, reward, next_state, done, dist)

    def clear_episode(self):
        self.episode_buffer.clear()

    def add_to_buffer(self):
        self.buffer.extend(self.episode_buffer.obtain_episode())

    def get_from_episode(self) -> dict:
        return self.episode_buffer.obtain_dict()

    def sample_from_buffer(self) -> dict:
        minibatch = random.sample(self.buffer, self._batchsize)
        states, actions, rewards, next_states, dones, dists = zip(*minibatch)
        batch_dict = {"state": np.array(states),
                      "action": actions,
                      "reward": rewards,
                      "next_state": np.array(next_states),
                      "done": dones}

        if self.episode_buffer.requires_distribution:
            batch_dict["dist"] = dists

        return batch_dict

    def buffer_size(self):
        return len(self.buffer)


class HindsightReplayBuffer:
    def __init__(self, max_size,
                 state_is_dict=False,
                 dtype=DTYPE, device=DEVICE):

        self.max_size = max_size
        self.state_is_dict = state_is_dict
        self.dtype = dtype
        self.device = device

        self.buffer = deque(maxlen=max_size)
        self.episode_buffer = deque()

    def add(self, state, action, reward, next_state, done):
        experience = [state, action, reward, next_state, done]
        self.episode_buffer.append(experience)

    def clear(self):
        self.episode_buffer.clear()

    def add_to_buffer(self, eb: deque):
        self.buffer.extend(eb)

    def sample(self, batch_size):
        indices = np.random.randint(len(self.buffer), size=batch_size)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])

        def to_tensor(data):
            return torch.tensor(data, dtype=self.dtype, device=self.device)

        def to_tensor_dict(data):
            return {key: to_tensor(value) for key, value in data.items()}

        if self.state_is_dict:
            return (states,
                    actions,
                    rewards,
                    next_states,
                    dones)
        else:
            return (to_tensor(states),
                    to_tensor(actions),
                    to_tensor(rewards),
                    to_tensor(next_states),
                    to_tensor(dones))

    def __len__(self):
        return len(self.buffer)
