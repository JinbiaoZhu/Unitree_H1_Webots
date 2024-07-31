"""
PPO algorithm.
"""
import torch.optim
import torch.nn.functional as F

from base_declaration import DEVICE, DTYPE, EPS, LINEAR
from buffer import Episode
from continuous_policy import ContinuousPolicyNormalLSTM
from continuous_critic import ContinuousValueCriticLSTM
from tools import init_weight, time_string


class PomdpPPO:
    def __init__(self, state_dim, action_dim, hidden_dim, recurrent_layers, max_action,
                 actor_lr, critic_lr, gamma, lmbda, epochs, epsilon,
                 is_trainable_std_dev, init_log_std_dev,
                 linear=LINEAR,
                 device=DEVICE, dtype=DTYPE, eps=EPS):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.hidden_dim, self.recurrent_layers = hidden_dim, recurrent_layers
        self.max_action = max_action
        self.actor_lr, self.critic_lr = actor_lr, critic_lr
        self.gamma, self.lmbda, self.epochs, self.epsilon = gamma, lmbda, epochs, epsilon
        self.is_trainable_std_dev, self.init_log_std_dev = is_trainable_std_dev, init_log_std_dev
        self.linear = linear
        self.device, self.dtype, self.eps = device, dtype, eps

        self.policy = ContinuousPolicyNormalLSTM(self.state_dim, self.action_dim, self.hidden_dim,
                                                 self.recurrent_layers, self.is_trainable_std_dev,
                                                 self.init_log_std_dev, self.dtype, self.device).to(self.device)
        self.policy = init_weight(self.policy)
        self.critic = ContinuousValueCriticLSTM(self.state_dim, self.hidden_dim, self.recurrent_layers, self.device).to(
            self.device)
        self.critic = init_weight(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_lr)

        self.replay_buffer = Episode(requires_distribution=False, requires_goal=False)

        print("PPO: state_dim={}, action_dim={}, hidden_dim={}, actor_lr={}, critic_lr={}, gamma={}".format(
            self.state_dim, self.action_dim, self.hidden_dim, self.actor_lr, self.critic_lr, self.gamma))

        print("lmbda={}, epsilon={}, epochs={}".format(self.lmbda, self.epsilon, self.epochs))

    def action(self, state):
        if len(state.shape) != 3:
            state = torch.as_tensor(state, dtype=self.dtype, device=self.device).reshape((1, 1, -1))
        else:
            state = torch.as_tensor(state, dtype=self.dtype, device=self.device)
        action_distribution, _, _ = self.policy(state)
        action = action_distribution.rsample()
        return action.squeeze(0).data.cpu().numpy(), None

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
        states = torch.tensor(sp["state"], dtype=self.dtype, device=self.device).view((1, -1, self.state_dim))
        actions = torch.tensor(sp["action"], dtype=self.dtype, device=self.device).view((1, -1, self.action_dim))
        next_states = torch.tensor(sp["next_state"], dtype=self.dtype, device=self.device).view((1, -1, self.state_dim))
        dones = torch.tensor(sp["done"], dtype=self.dtype, device=self.device).view((1, -1, 1))
        rewards = torch.tensor(sp["reward"], dtype=self.dtype, device=self.device).view((1, -1, 1))

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        _, loc, var = self.policy(states)
        old_action_dists = torch.distributions.MultivariateNormal(loc, var)
        old_log_probs = old_action_dists.log_prob(actions)

        actor_loss_total, critic_loss_total = 0, 0

        for i in range(self.epochs):
            _, loc, var = self.policy(states)
            action_dists = torch.distributions.MultivariateNormal(loc, var)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = advantage * ratio
            surr2 = advantage * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            actor_loss = -1 * torch.mean(torch.min(surr1, surr2))
            left = self.critic(states)
            right = td_target.squeeze(0).detach()
            critic_loss = torch.mean(F.mse_loss(left, right))

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
