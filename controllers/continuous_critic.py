"""
Continuous critics, used for DDPG, PPO.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ContinuousValueCriticLSTM(nn.Module):
    def __init__(self, state_dim, hidden_size, recurrent_layers, device):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_size, self.recurrent_layers = hidden_size, recurrent_layers
        self.device = device

        self.layer_lstm = nn.LSTM(self.state_dim, self.hidden_size, num_layers=self.recurrent_layers).to(self.device)
        self.layer_hidden = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
        self.layer_value = nn.Linear(self.hidden_size, 1).to(self.device)
        self.hidden_cell = None

    def get_init_state(self, batch_size):
        return (torch.zeros(self.recurrent_layers, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.recurrent_layers, batch_size, self.hidden_size).to(self.device))

    def forward(self, state, hidden_state):
        batch_size = state.shape[1]
        if hidden_state is None or batch_size != hidden_state[0].shape[1]:
            hidden_state = self.get_init_state(batch_size)

        _, new_hidden_state = self.layer_lstm(state, hidden_state)
        hidden_out = F.elu(self.layer_hidden(new_hidden_state[0][-1]))
        value_out = self.layer_value(hidden_out)
        return value_out


if __name__ == "__main__":
    # This is just a simple toy case
    lstm_policy = ContinuousValueCriticLSTM(state_dim=21, hidden_size=324, recurrent_layers=2, device="cuda:0")

    state_1 = torch.rand(100, 1, 21).to(device="cuda:0")
    output = lstm_policy(state_1)
    print(output)

    state_2 = torch.rand(50, 1, 21).to(device="cuda:0")
    output = lstm_policy(state_2)
    print(output)

    state_3 = torch.rand(1, 64, 21).to(device="cuda:0")
    output = lstm_policy(state_3)
    print(output.shape)
