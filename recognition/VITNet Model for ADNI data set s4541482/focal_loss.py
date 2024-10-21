# focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss with support for class-wise alpha.
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Args:
            alpha (list, tuple, float, int, or torch.Tensor, optional): Weighting factor for each class.
                - If None, no weighting is applied.
                - If list or tuple, it should contain weights for each class.
                - If float or int, it's treated as the weight for the positive class.
                - If torch.Tensor, it should contain weights for each class.
            gamma (float): Focusing parameter to reduce the loss for well-classified examples.
            reduction (str): Reduction method ('mean', 'sum', 'none').
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = torch.tensor(1.0)
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        else:
            raise TypeError('Unsupported type for alpha. Must be float, int, list, tuple, or torch.Tensor.')

        # Register alpha as a buffer to ensure it's moved to the correct device
        self.register_buffer('alpha_buffer', self.alpha)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted logits [batch_size, num_classes].
            targets (torch.Tensor): Ground truth labels [batch_size].
        """
        if inputs.dim() > 2:
            inputs = inputs.view(-1, inputs.size(-1))  # Reshape if necessary

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # [batch_size]
        pt = torch.exp(-ce_loss)  # Probability of the true class

        # Handle different types of alpha
        if self.alpha_buffer.dim() == 1:
            alpha = self.alpha_buffer[targets]
        else:
            alpha = self.alpha_buffer

        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
