"""
Contains the code for calculating losses and metrics for the Unet, as well as functions for saving them

@author Carl Flottmann
"""
from __future__ import annotations
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as functional


class Accuracy():
    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def forward(self, prediction: torch.Tensor, truth: torch.Tensor) -> None:
        prediction_mask = torch.argmax(prediction, dim=1)
        self.correct += (prediction_mask == truth).sum().item()
        self.total += truth.numel()

    def accuracy(self) -> float:
        return 100 * self.correct / self.total


class DiceLoss(_Loss):
    SMOOTH_FACTOR = 1e-7

    def __init__(self, num_classes: int, device: torch.device, smooth_factor: float = SMOOTH_FACTOR) -> None:
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth_factor = smooth_factor
        self.device = device

        # will be a tensor on first use, stores each iteration of loss tracking
        self.epoch_loss_tracking = None
        # stores the loss for each iteration for each class, including the total loss
        self.iteration_loss_tracking = torch.zeros(
            0, self.num_classes + 1, device=self.device)

    def get_smooth(self) -> float:
        return self.smooth_factor

    def reset(self) -> None:
        self.epoch_loss_tracking = None
        self.reset_epoch()

    def reset_epoch(self) -> None:
        self.iteration_loss_tracking = torch.zeros(
            0, self.num_classes + 1, device=self.device)

    def epochs_run(self) -> int:
        return len(self.epoch_loss_tracking[0])

    def iterations_run(self) -> int:
        return len(self.iteration_loss_tracking[0])

    def forward(self, prediction: torch.Tensor, truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = prediction.to(self.device)
        truth = truth.to(self.device)

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

        # Compute loss for each class
        for i in range(self.num_classes):
            intersection = (prediction_flat[:, i]
                            * truth_flat[:, i]).sum(dim=-1)
            dice = (2. * intersection + self.smooth_factor) / (
                prediction_flat[:, i].sum(
                    dim=-1) + truth_flat[:, i].sum(dim=-1) + self.smooth_factor
            )

            # Dice coefficient is dice.mean(), so loss is 1 - dice.mean()
            class_losses.append(1 - dice.mean())

        total_loss = torch.mean(torch.stack(class_losses))

        # store the current losses in our tracking tensor
        iteration_losses = torch.stack(
            [total_loss] + class_losses).unsqueeze(0).to(self.device)
        self.iteration_loss_tracking = torch.cat(
            (self.iteration_loss_tracking, iteration_losses), dim=0)

        return total_loss, class_losses

    def save_epoch(self) -> None:
        if self.epoch_loss_tracking is None:
            self.epoch_loss_tracking = self.iteration_loss_tracking.unsqueeze(
                0)
        else:
            self.epoch_loss_tracking = torch.cat(
                (self.epoch_loss_tracking, self.iteration_loss_tracking.unsqueeze(0)), dim=0)

        self.reset_epoch()

    def get_all_losses(self) -> list[list[float]]:
        losses = [[] for _ in range(self.num_classes + 1)]

        if self.epoch_loss_tracking is None:  # haven't called save_epoch() yet
            if (torch.all(self.iteration_loss_tracking == 0)):  # haven't called forward() yet
                raise AttributeError(
                    "No data has been passed through the loss function yet")
            else:  # have run through some iterations, return that
                for i in range(self.num_classes + 1):
                    class_losses = self.iteration_loss_tracking[:, i]
                    for loss in class_losses:
                        losses[i].append(loss.item())

        else:  # have run through some epochs and iterations
            for epoch in self.epoch_loss_tracking:
                for i in range(self.num_classes + 1):
                    class_losses = epoch[:, i]  # get the losses for class i
                    for loss in class_losses:
                        losses[i].append(loss.item())

        return losses

    def get_average_losses(self) -> list[list[float]]:
        losses = [[] for _ in range(self.num_classes + 1)]

        if self.epoch_loss_tracking is None:  # haven't called save_epoch() yet
            if (torch.all(self.iteration_loss_tracking == 0)):  # haven't called forward() yet
                raise AttributeError(
                    "No data has been passed through the loss function yet")
            else:  # have run through some iterations, return that
                for i in range(self.num_classes + 1):
                    avg_loss = torch.mean(
                        self.iteration_loss_tracking[:, i]).item()
                    losses[i].append(avg_loss)

        else:  # have run through some epochs and iterations
            for epoch in self.epoch_loss_tracking:
                for i in range(self.num_classes + 1):
                    avg_loss = torch.mean(epoch[:, i]).item()
                    losses[i].append(avg_loss)

        return losses

    def get_end_losses(self) -> list[list[float]]:
        losses = [[] for _ in range(self.num_classes + 1)]

        if self.epoch_loss_tracking is None:  # haven't called save_epoch() yet
            if (torch.all(self.iteration_loss_tracking == 0)):  # haven't called forward() yet
                raise AttributeError(
                    "No data has been passed through the loss function yet")
            else:  # have run through some iterations, return that
                for i in range(self.num_classes + 1):
                    end_loss = self.iteration_loss_tracking[-1, i].item()
                    losses[i].append(end_loss)

        else:  # have run through some epochs and iterations
            for epoch in self.epoch_loss_tracking:
                for i in range(self.num_classes + 1):
                    end_loss = epoch[-1, i].item()
                    losses[i].append(end_loss)

        return losses

    def get_all_losses_tensor(self) -> torch.Tensor:
        if self.epoch_loss_tracking is None:
            if (torch.all(self.iteration_loss_tracking == 0)):
                raise AttributeError(
                    "No data has been passed through the loss function yet")
            else:
                return self.iteration_loss_tracking
        return self.epoch_loss_tracking

    def get_average_losses_tensor(self) -> torch.Tensor:
        if self.epoch_loss_tracking is None:
            if (torch.all(self.iteration_loss_tracking == 0)):
                raise AttributeError(
                    "No data has been passed through the loss function yet")
            else:
                return torch.mean(self.iteration_loss_tracking, dim=1)
        return torch.mean(self.epoch_loss_tracking, dim=1)

    def get_end_losses_tensor(self) -> torch.Tensor:
        if self.epoch_loss_tracking is None:
            if (torch.all(self.iteration_loss_tracking == 0)):
                raise AttributeError(
                    "No data has been passed through the loss function yet")
            else:
                return self.iteration_loss_tracking[:, -1, :]
        return self.epoch_loss_tracking[:, -1, :]

    def state_dict(self) -> dict[str: list[list[torch.Tensor]]]:
        return {
            'epoch_loss_tracking': self.epoch_loss_tracking,
            'iteration_loss_tracking': self.iteration_loss_tracking,
            'num_classes': self.num_classes,
            'smooth_factor': self.smooth_factor,
            'device': self.device,
        }

    @staticmethod
    def load_state_dict(state_dict: dict[str: list[list[torch.Tensor]]]) -> DiceLoss:
        obj = DiceLoss(state_dict['num_classes'], state_dict['device'],
                       smooth_factor=state_dict['smooth_factor'])
        obj.epoch_loss_tracking = state_dict['epoch_loss_tracking']
        obj.iteration_loss_tracking = state_dict['iteration_loss_tracking']
        return obj
