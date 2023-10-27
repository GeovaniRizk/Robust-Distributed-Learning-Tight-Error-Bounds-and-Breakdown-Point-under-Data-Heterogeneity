import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


class logistic(torch.nn.Module):
    """ Simple, small fully connected model.
    """

    def __init__(self, nb_classes):
        """ Model parameter constructor.
        """
        super().__init__()
        # Build parameters
        self._f1 = torch.nn.Linear(28 * 28, nb_classes)

    def forward(self, x):
        """ Model's forward pass.
        Args:
            x Input tensor
        Returns:
            Output tensor
        """
        # Forward pass
        x = self._f1(x.view(-1, 28 * 28))
        return x

