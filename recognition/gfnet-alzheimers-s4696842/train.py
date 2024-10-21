import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from .modules import GFNet
from .dataset import get_train_test_dataloaders, ADNI_IMAGE_DIMENSIONS
from .utils import ADNI_CLASSES, get_device

""" 
Hyperparamters 
"""
learning_rate = 1e-3
weight_decay = 1e-2
T_max = 6
depth = 12
embed_dim = 260
drop_rate = 0.1
drop_path_rate = 0.1
patch_size = (8, 8)


def train(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=10,
    test=True,
):
    """
    Trains the given model on the training dataset and optionally evaluates it on the test dataset.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        device (torch.device): Device to run the training on ('cpu', 'cuda', etc.).
        num_epochs (int, optional): Number of epochs to train. Defaults to 10.
        test (bool, optional): If True, evaluates the model on the test dataset after each epoch. Defaults to True.

    Returns:
        tuple:
            - train_loss_history (list of float): Training loss recorded after each epoch.
            - train_acc_history (list of float): Training accuracy recorded after each epoch.
            - test_loss_history ((list of float) or None): Test loss recorded after each epoch if `test` is True; otherwise None.
            - test_acc_history ((list of float) or None): Test accuracy recorded after each epoch if `test` is True; otherwise None.
    """
    train_loss_history = []
    train_acc_history = []
    test_loss_history = [] if test else None
    test_acc_history = [] if test else None

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        n_batches = len(train_loader)

        model.train()
        for batch, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print(
                f"Batch {batch}/{n_batches}, Cum. Loss: {epoch_loss/(batch+1)}, Cum. Accuracy: {correct/total}"
            )

        scheduler.step()

        epoch_loss = epoch_loss / n_batches
        epoch_acc = correct / total

        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}"
        )

        if test:
            test_loss, test_acc = run_test(model, test_loader, criterion, device)
            test_loss_history.append(test_loss)
            test_acc_history.append(test_acc)

    return train_loss_history, train_acc_history, test_loss_history, test_acc_history


def run_test(model, test_loader, criterion, device):
    """
    Evaluates the model on the test dataset.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function used during training.
        device (torch.device): Device to run the evaluation on ('cpu', 'cuda', etc.).

    Returns:
        tuple:
            - avg_test_loss (float): Average loss over the test dataset.
            - test_accuracy (float): Accuracy over the test dataset.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = correct / total

    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return avg_test_loss, test_accuracy


# Plotting function
def plot_metrics(
    train_loss_history, train_acc_history, test_loss_history=None, test_acc_history=None
):
    """
    Plots training and test loss and accuracy over epochs.

    Args:
        train_loss_history (list of float): List of training loss values per epoch.
        train_acc_history (list of float): List of training accuracy values per epoch.
        test_loss_history (list of float, optional): List of test loss values per epoch. Defaults to None.
        test_acc_history (list of float, optional): List of test accuracy values per epoch. Defaults to None.

    Returns:
        None
    """
    # Import is done here so matplotlib is only a requirement if plotting is
    # required.
    from matplotlib import pyplot as plt

    epochs = range(1, len(train_loss_history) + 1)

    plt.figure(figsize=(14, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_history, label="Train Loss")
    if test_loss_history:
        plt.plot(epochs, test_loss_history, label="Test Loss", linestyle="--")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_history, label="Train Accuracy")
    if test_acc_history:
        plt.plot(epochs, test_acc_history, label="Test Accuracy", linestyle="--")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()


def get_optimizer(model):
    """
    Initializes the optimizer and learning rate scheduler for the model.

    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.

    Returns:
        tuple:
            - optimizer (torch.optim.Optimizer): The optimizer instance.
            - scheduler (torch.optim.lr_scheduler.CosineAnnealingLR): The learning rate scheduler instance.
    """
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    return optimizer, scheduler


def main(model, train_loader, test_loader, num_epochs=10, test=True, plot=True):
    """
    Runs the training and evaluation pipeline.

    Args:
        model (torch.nn.Module): The model to train and evaluate.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        num_epochs (int, optional): Number of epochs for training. Defaults to 10.
        test (bool, optional): If True, evaluates the model on the test dataset after each epoch. Defaults to True.
        plot (bool, optional): If True, plots the training and test metrics after training. Defaults to True.

    Returns:
        None
    """
    device = torch.device(get_device())
    print(f"device={get_device()}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer(model)

    train_loss_history, train_acc_history, test_loss_history, test_acc_history = train(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=num_epochs,
        test=test,
    )

    if plot:
        plot_metrics(
            train_loss_history, train_acc_history, test_loss_history, test_acc_history
        )
    else:
        print(train_loss_history)
        print(train_acc_history)
        print(test_loss_history)
        print(test_acc_history)


def parse_args():
    """
    Parses command-line arguments for configuring the model training and evaluation.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed command-line arguments, containing:
            - data (str): Path to the training and test data directories.
            - plot (bool): Whether to display a plot of metrics after training.
            - save_to (str): Path to save the trained model (default: 'model.pth').
            - epochs (int): Number of epochs to train the model (default: 2).
            - test (bool): Whether to evaluate the model on the test dataset.
            - save_error (bool): Whether to save the model in case of an error.
            - batch_size (int): Batch size for training and testing (default: 32).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data", type=str, help="Path to train and test data directories"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Display plot of metrics after training"
    )
    parser.add_argument(
        "--save-to",
        type=str,
        help="Path to save the trained model",
        default="model.pth",
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs to train the model", default=2
    )
    parser.add_argument(
        "--test", action="store_true", help="Evaluate the model on the test dataset"
    )
    parser.add_argument(
        "--save-error",
        action="store_true",
        help="Save the model if an error occurs during training",
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size for training and testing", default=32
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = GFNet(
        img_size=ADNI_IMAGE_DIMENSIONS,
        in_chans=1,
        num_classes=len(ADNI_CLASSES),
        depth=depth,
        embed_dim=embed_dim,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        patch_size=patch_size,
    )
    train_loader, test_loader = get_train_test_dataloaders(
        root_dir=args.data,
        train_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
    )
    try:
        print(
            f"Starting training loop: root_dir={args.data}, batch_size={args.batch_size}, epochs={args.epochs}, test={args.test}, plot={args.plot}"
        )
        main(
            model,
            train_loader,
            test_loader,
            num_epochs=args.epochs,
            test=args.test,
            plot=args.plot,
        )
        torch.save(model.state_dict(), args.save_to)
    except:
        if args.save_error:
            torch.save(model.state_dict(), "error_" + args.save_to)
        raise
