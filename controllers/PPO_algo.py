"""
PPO algorithm.
"""
import copy

import numpy as np
import torch.optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters

from base_declaration import DEVICE, DTYPE, EPS, LINEAR
from buffer import Episode
from continuous_policy import ContinuousPolicyNormal
from continuous_critic import ContinuousValueCritic
from tools import init_weight, discounted_rewards, time_string


class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim, max_action,
                 actor_lr, critic_lr, gamma, lmbda, epochs, epsilon,
                 linear=LINEAR,
                 device=DEVICE, eps=EPS):
        self.state_dim, self.action_dim, self.hidden_dim = state_dim, action_dim, hidden_dim
        self.max_action = max_action

        self.actor_lr, self.critic_lr = actor_lr, critic_lr
        self.gamma, self.lmbda, self.epsilon = gamma, lmbda, epsilon

        self.epochs = epochs
        self.linear = linear

        self.device, self.eps = device, eps

        self.policy = ContinuousPolicyNormal(self.state_dim, self.action_dim, self.max_action, self.linear,
                                             self.hidden_dim, "relu").to(self.device)
        self.policy = init_weight(self.policy)
        self.critic = ContinuousValueCritic(self.state_dim, self.action_dim, self.max_action, self.hidden_dim,
                                            "relu").to(self.device)
        self.critic = init_weight(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_lr)

        self.replay_buffer = Episode(requires_distribution=False, requires_goal=False)

        print("PPO: state_dim={}, action_dim={}, hidden_dim={}, actor_lr={}, critic_lr={}, gamma={}".format(
            self.state_dim, self.action_dim, self.hidden_dim, self.actor_lr, self.critic_lr, self.gamma))

        print("lmbda={}, epsilon={}, epochs={}".format(self.lmbda, self.epsilon, self.epochs))

    def action(self, state):
        state = torch.as_tensor(state, dtype=DTYPE, device=self.device)
        # No deterministic
        mean, logstd = self.policy(state)
        dist = Normal(mean, logstd.exp())
        action = dist.sample()
        return action.data.cpu().numpy(), None

    def store(self, state, action, reward, next_state, done, dist):
        self.replay_buffer.add(state, action, reward, next_state, done, dist)

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def update(self):
        # Get datas
        sp = self.replay_buffer.obtain_dict()
        # Data transition to tensor
        states = torch.tensor(sp["state"], dtype=DTYPE, device=self.device).view((-1, self.state_dim))
        actions = torch.tensor(sp["action"], dtype=DTYPE, device=self.device).view((-1, self.action_dim))
        next_states = torch.tensor(sp["next_state"], dtype=DTYPE, device=self.device).view((-1, self.state_dim))
        dones = torch.tensor(sp["done"], dtype=DTYPE, device=self.device).view((-1, 1))
        rewards = torch.tensor(sp["reward"], dtype=DTYPE, device=self.device).view((-1, 1))

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(self.gamma, self.lmbda,
                                           td_delta.cpu()).to(self.device)
        mu, log_std = self.policy(states)
        old_action_dists = torch.distributions.Normal(mu.detach(),
                                                      log_std.exp().detach())
        old_log_probs = old_action_dists.log_prob(actions)

        actor_loss_total, critic_loss_total = 0, 0

        for i in range(self.epochs):

            mu, log_std = self.policy(states)
            action_dists = Normal(mu, log_std.exp())
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs-old_log_probs)
            surr1 = advantage*ratio
            surr2 = advantage*torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
            actor_loss = -1*torch.mean(torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            actor_loss_total += actor_loss.data.item()
            critic_loss_total += critic_loss.data.item()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return actor_loss_total, critic_loss_total

    def save(self, filedir):

        total_filepath = filedir + "actor_" + time_string() + ".pth"
        print(f"Saving actor model to {total_filepath} ...")
        torch.save(self.policy, total_filepath)
        print("Successfully!")

