"""
Continuous policies, used for REINFORCE, .
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
