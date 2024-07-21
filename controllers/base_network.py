"""
This base network used for other policies.
"""

import torch
import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

    def save(self, filename: str, path: str):
        torch.save(self.state_dict(), filename, path)
        pass

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
        pass
