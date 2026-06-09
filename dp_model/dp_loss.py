"""Official KL divergence loss (UKBiobank_deep_pretrain)."""

import torch.nn as nn


def my_KLDivLoss(x, y):
    """K-L divergence: sum reduction, averaged over batch; y += 1e-16."""
    loss_func = nn.KLDivLoss(reduction="sum")
    y = y + 1e-16
    n = y.shape[0]
    return loss_func(x, y) / n
