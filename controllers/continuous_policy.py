"""
Continuous policies, used for REINFORCE, PPO.
"""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from base_declaration import *
from base_network import BaseNet


class ContinuousPolicyNormal(BaseNet):
    """
    Used for REINFORCE, AC,
    """

    def __init__(self, state_dim, action_dim, max_action, linear=1.0, hidden_dim=256, activate="tanh",
                 log_std_max=LOG_STD_MAX, log_std_min=LOG_STD_MIN, eps=EPS, device=DEVICE):
        super().__init__()

        self.max_action = max_action
        self.linear = linear
        self.log_std_max, self.log_std_min = log_std_max, log_std_min
        self.eps = eps
        self.device = device

        if activate in ["relu", "tanh"]:
            if activate == "relu":
                self.activate = nn.ReLU()
            elif activate == "tanh":
                self.activate = nn.Tanh()
            else:
                raise ValueError("Unknown activation function.")

        self.q1 = nn.Linear(state_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, hidden_dim)
        self.q3_mean = nn.Linear(hidden_dim, action_dim)
        self.q3_logstd = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        if isinstance(x, dict):
            x = np.concatenate((x["observation"], x["desired_goal"]))
            x = torch.as_tensor(x, dtype=DTYPE, device=self.device).view((1, -1))

        x = self.activate(self.q1(x))
        x = self.activate(self.q2(x))
        mean = torch.tanh(self.q3_mean(x))
        logstd = F.leaky_relu(self.q3_logstd(x))
        logstd = torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)
        return mean * self.linear, logstd * (self.linear ** 2)


class ContinuousPolicyDeterministic(BaseNet):
    """
    Used in DDPG
    """

    def __init__(self, state_dim, action_dim, max_action,
                 hidden_dim=256, activate="relu", device=DEVICE):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.hidden_dim = hidden_dim

        if activate in ["relu", "tanh"]:
            if activate == "relu":
                self.activate = nn.ReLU()
            elif activate == "tanh":
                self.activate = nn.Tanh()
        else:
            raise ValueError("Unknown activation function.")

        self.device = device

        self.p1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.p2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.p3 = nn.Linear(self.hidden_dim, self.action_dim)

    def forward(self, x):
        if isinstance(x, dict):
            x = np.concatenate((x["observation"], x["desired_goal"]))
            x = torch.as_tensor(x, dtype=DTYPE, device=self.device).view((1, -1))

        x = self.activate(self.p1(x))
        x = self.activate(self.p2(x))
        x = torch.tanh(self.p3(x))
        x = x * self.max_action
        return x


class ContinuousPolicyNormalLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, recurrent_layers, is_trainable_std_dev, init_log_std_dev,
                 dtype, device):
        super().__init__()

        self.state_dim, self.action_dim = state_dim, action_dim
        self.hidden_size, self.recurrent_layers = hidden_size, recurrent_layers
        self.is_trainable_std_dev, self.init_log_std_dev = is_trainable_std_dev, init_log_std_dev
        self.dtype, self.device = dtype, device

        self.lstm = nn.LSTM(self.state_dim, self.hidden_size, num_layers=self.recurrent_layers).to(device=self.device)
        self.layer_hidden = nn.Linear(self.hidden_size, self.hidden_size).to(device=self.device)
        self.layer_policy_logits = nn.Linear(self.hidden_size, self.action_dim).to(device=self.device)
        self.log_std_dev = nn.Parameter(
            self.init_log_std_dev * torch.ones(self.action_dim, dtype=self.dtype).unsqueeze(0),
            requires_grad=self.is_trainable_std_dev).to(device=self.device)
        self.covariance_eye = torch.eye(self.action_dim).unsqueeze(0).to(device=self.device)

        self.hidden_cell = None

    def get_init_state(self, batch_size):
        self.hidden_cell = (torch.zeros(self.recurrent_layers, batch_size, self.hidden_size).to(self.device),
                            torch.zeros(self.recurrent_layers, batch_size, self.hidden_size).to(self.device))

    def forward(self, state):
        batch_size = state.shape[1]
        device = state.device
        if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(batch_size)
        self.hidden_cell = [value for value in self.hidden_cell]
        _, self.hidden_cell = self.lstm(state, self.hidden_cell)
        hidden_out = F.elu(self.layer_hidden(self.hidden_cell[0][-1]))
        policy_logits_out = self.layer_policy_logits(hidden_out)
        cov_matrix = self.covariance_eye.to(device).expand(batch_size, self.action_dim,
                                                           self.action_dim) * torch.exp(self.log_std_dev.to(device))
        # We define the distribution on CPU since otherwise operations fail with CUDA illegal memory access error.
        policy_dist = torch.distributions.multivariate_normal.MultivariateNormal(policy_logits_out.to("cpu"),
                                                                                 cov_matrix.to("cpu"))
        return policy_dist, policy_logits_out, cov_matrix


if __name__ == "__main__":
    # This is just a simple toy case
    lstm_policy = ContinuousPolicyNormalLSTM(state_dim=21, action_dim=24,
                                             hidden_size=324, recurrent_layers=2,
                                             is_trainable_std_dev=True, init_log_std_dev=0.0,
                                             dtype=torch.float32, device="cuda:0")

    state_1 = torch.rand(100, 1, 21).to(device="cuda:0")
    output = lstm_policy(state_1)
    print(output)

    state_2 = torch.rand(50, 1, 21).to(device="cuda:0")
    output = lstm_policy(state_2)
    print(output)

    state_3 = torch.rand(1, 64, 21).to(device="cuda:0")
    output = lstm_policy(state_3)
    print(output)
