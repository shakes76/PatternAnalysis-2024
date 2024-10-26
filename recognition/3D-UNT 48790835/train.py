import os
import dataset
import modules
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

# Hyper-parameters
num_epochs = 30
learning_rate = 5 * 10 ** -4
batch_size = 4
learning_rate_decay = 0.985


validation_images_path = "/Users/qiuhan/Desktop/UQ/3710/Improved-UNET-s4879083/重新下载的数据集/Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/data/HipMRI_study_complete_release_v1/semantic_MRs_anon"
train_images_path = "/Users/qiuhan/Desktop/UQ/3710/Improved-UNET-s4879083/重新下载的数据集/Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/data/HipMRI_study_complete_release_v1/semantic_MRs_anon"
validation_labels_path = "/Users/qiuhan/Desktop/UQ/3710/Improved-UNET-s4879083/重新下载的数据集/Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/data/HipMRI_study_complete_release_v1/semantic_labels_anon"
train_labels_path = "/Users/qiuhan/Desktop/UQ/3710/Improved-UNET-s4879083/重新下载的数据集/Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-/data/HipMRI_study_complete_release_v1/semantic_labels_anon"
model_path = "3d_unet_model.pt"



def init():
    transform = dataset.get_transform()
    valid_dataset = dataset.Medical3DDataset(validation_images_path, validation_labels_path, transform=transform)
    train_dataset = dataset.Medical3DDataset(train_images_path, train_labels_path, transform=transform)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning: CUDA not found. Using CPU")

    data_loaders = {'train': train_loader, 'valid': valid_loader}
    return data_loaders, device


def main():
    data_loaders, device = init()
    model = modules.Improved3DUnet()
    model = model.to(device)
    train_and_validate(data_loaders, model, device)
    torch.save(model.state_dict(), model_path)


def train_and_validate(data_loaders, model, device):
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay)

    train_losses, train_dice, val_losses, val_dice = [], [], [], []

    for epoch in range(num_epochs):
        train_loss, train_coeff = train(data_loaders["train"], model, device, criterion, optimizer)
        val_loss, val_coeff = validate(data_loaders["valid"], model, device, criterion)

        train_losses.append(train_loss)
        train_dice.append(train_coeff)
        val_losses.append(val_loss)
        val_dice.append(val_coeff)

        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.5f}, Training Dice: {train_coeff:.5f}")
        print(f"Validation Loss: {val_loss:.5f}, Validation Dice: {val_coeff:.5f}")

    save_plot(train_losses, val_losses, "Loss", "LossCurve.png")
    save_plot(train_dice, val_dice, "Dice Coefficient", "DiceCurve.png")


def train(loader, model, device, criterion, optimizer):
    model.train()
    losses, dice_scores = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        losses.append(loss.item())

        dice_score = dice_coefficient(outputs, labels).item()
        dice_scores.append(dice_score)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(losses), np.mean(dice_scores)


def validate(loader, model, device, criterion):
    model.eval()
    losses, dice_scores = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            losses.append(loss.item())

            dice_score = dice_coefficient(outputs, labels).item()
            dice_scores.append(dice_score)

    return np.mean(losses), np.mean(dice_scores)


def dice_loss(outputs, labels):
    return 1 - dice_coefficient(outputs, labels)


def dice_coefficient(outputs, labels, epsilon=1e-8):
    intersection = (outputs * labels).sum()
    return (2. * intersection) / ((outputs + labels).sum() + epsilon)


def save_plot(train_list, val_list, metric, path):
    epochs = list(range(1, num_epochs + 1))
    plt.plot(epochs, train_list, label=f"Training {metric}")
    plt.plot(epochs, val_list, label=f"Validation {metric}")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.title(f"Training and Validation {metric} Over Epochs")
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main()
