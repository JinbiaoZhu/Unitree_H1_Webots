"""
Continuous critics, used for DDPG, PPO.
"""
import torch.nn as nn

from base_network import BaseNet


class ContinuousCritic(BaseNet):
    """
    Q(s,a)
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, activate="relu"):
        super(ContinuousCritic, self).__init__()
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

        self.c1 = nn.Linear(self.state_dim + self.action_dim, self.hidden_dim)
        self.c2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.c3 = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        x = self.activate(self.c1(x))
        x = self.activate(self.c2(x))
        x = self.c3(x)
        return x


class ContinuousValueCritic(BaseNet):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, activate="relu"):
        super(ContinuousValueCritic, self).__init__()

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

        self.c1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.c2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.c3 = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        x = self.activate(self.c1(x))
        x = self.activate(self.c2(x))
        x = self.c3(x)
        return x
