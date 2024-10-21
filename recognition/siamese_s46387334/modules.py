"""
Contains the source code for the components of the Siamese Net and Classifier.

Each component is implementated as a class or a function.
"""

###############################################################################
### Imports
import torch
import torch.nn as nn




###############################################################################
### Classes
class TripletLoss(nn.Module):
    """
    """
    def __init__(self, margin: float=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def euclidean_dist(self, x1: torch.Tensor, x2: torch.Tensor) -> float:
        """
        """
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        """
        distance_positive = self.euclidean_dist(anchor, positive)
        distance_negative = self.euclidean_dist(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()