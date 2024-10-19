"""
Contains the code for calculating losses and metrics for the Unet

@author Carl Flottmann
"""
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as functional


class DiceLoss(_Loss):
    SMOOTH_FACTOR = 1e-7

    def __init__(self, num_classes: int, smooth_factor: int = SMOOTH_FACTOR) -> None:
        super(DiceLoss, self).__init__()

        self.num_classes = num_classes
        self.smooth_factor = smooth_factor

    def forward(self, prediction: torch.Tensor, truth: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch_size = truth.size(0)
        num_classes = prediction.size(1)

        # one-hot encode mask, (N, H, W) to (N, C, H, W)
        truth = functional.one_hot(truth.squeeze(
            1).view(-1), num_classes).view(batch_size, -1, self.num_classes).permute(0, 2, 1)

        # Flatten tensors to (N, C, H*W) for easy spatial computation
        prediction_flat = prediction.view(
            prediction.size(0), self.num_classes, -1)
        truth_flat = truth.view(truth.size(0), self.num_classes, -1)

        class_losses = []

        # compute loss for each class
        for i in range(self.num_classes):
            intersection = (prediction_flat[:, i]
                            * truth_flat[:, i]).sum(dim=-1)
            dice = (2. * intersection + self.smooth_factor) / \
                   (prediction_flat[:, i].sum(
                       dim=-1) + truth_flat[:, i].sum(dim=-1) + self.smooth_factor)

            # dice coefficient is dice.mean(), so loss is 1 - dice.mean()
            class_losses.append(1 - dice.mean())

        total_loss = torch.mean(torch.stack(class_losses))
        return total_loss, class_losses
