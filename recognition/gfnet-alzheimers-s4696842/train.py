import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from .modules import GFNet
from .dataset import get_train_test_dataloaders, ADNI_IMAGE_DIMENSIONS

learning_rate = 1e-3
weight_decay = 1e-2
accuracy_threshold = 0.8


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
            if test_acc > accuracy_threshold:
                torch.save(model.state_dict(), "over_80.pth")
                print("Model got over 80. Saved.")
                break

    return train_loss_history, train_acc_history, test_loss_history, test_acc_history


def run_test(model, test_loader, criterion, device):
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


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_optimizer(model):
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    return optimizer, scheduler


def main(model, train_loader, test_loader, num_epochs=10, test=True, plot=True):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data", type=str, help="Path to train and test data directories"
    )
    parser.add_argument("--plot", action="store_true", help="Display plot")
    parser.add_argument(
        "--save-to", type=str, help="Path to save model output", default="model.pth"
    )
    parser.add_argument(
        "--epochs", type=int, help="Path to save model output", default=2
    )
    parser.add_argument("--test", action="store_true", help="Path to save model output")
    parser.add_argument(
        "--batch-size", type=int, help="Path to save model output", default=32
    )
    args = parser.parse_args()

    height, width = ADNI_IMAGE_DIMENSIONS
    model = GFNet(
        img_size=height * width,
        in_chans=1,
        num_classes=2,
        depth=4,
        embed_dim=32,
        drop_rate=0.1,
        drop_path_rate=0.1,
        patch_size=16,
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
        torch.save(model.state_dict(), "error_" + args.save_to)
        raise
