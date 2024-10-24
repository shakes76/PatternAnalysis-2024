import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules import GFNet
from dataset import get_data_loaders
import os
import matplotlib.pyplot as plt

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the test and train dataset
    train_loader, test_loader = get_data_loaders(
        "D:/UQ Study/Y2 S2/COMP3710/Assesment/Report/PatternAnalysis-2024/recognition/ADNI_s4692111/AD_NC",
        batch_size=16,
    )
    # train_loader, test_loader = get_data_loaders("/home/groups/comp3710/ADNI/AD_NC/", batch_size=128)

    model = GFNet(img_size=224, patch_size=16, in_chans=1, num_classes=2, embed_dim=768, depth=12).to(device)
    
    # check if there is a saved model
    model_path = 'adni_model.pth'
    if os.path.exists(model_path):
        print(f"Load saved model：{model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Did not find saved model, start a new training.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # start training
    model.train()
    num_epochs = 100
    bar = tqdm(range(num_epochs * len(train_loader)), desc="train")

    # save every epoch's loss and accurate
    epoch_losses = []
    epoch_accuracies = []

    # loop train
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # calculate accurate
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            bar.update()

        scheduler.step()

        # calculate every epoch's loss and accurate
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)

        # print the loss and accuracy of epoch
        tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Save model
        torch.save(model.state_dict(), 'adni_model.pth')

    # draw the picture of loss and accurate
    plt.figure(figsize=(10,5))
    
    # draw Loss line
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), epoch_losses, label="Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    # draw Loss accuracy line
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), epoch_accuracies, label="Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("loss.png")

    return model, test_loader
