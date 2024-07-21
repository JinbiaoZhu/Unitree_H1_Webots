"""
Some declarations.
"""
import torch

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DEVICE1 = torch.device("cuda:1")
DTYPE = torch.float32
DTYPE_float = torch.float32
DTYPE_int = torch.int64
LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-7
LINEAR = 1.0

if __name__ == "__main__":
    """
    Test.
    """
    print(DEVICE)
    print(DTYPE)
