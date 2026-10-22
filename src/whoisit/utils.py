"""Small helpers shared by training, evaluation and scripts."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs) for reproducible runs.

    Args:
        seed: Integer seed, stored with every run in results/runs/<name>/.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
