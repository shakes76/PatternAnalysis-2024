import time
import torch
import torch.nn as nn
import os
import torch.optim as optim
from tqdm import tqdm

from modules import GFNet
from dataset import get_data_loaders
from utils import split_val_set, load_model
from predict import validate, final_validate


def train(model, train_loader, criterion, optimizer, device):
    """
    Train function. Return loss and accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


if __name__ == "__main__":
    # split the val set first, you only need to run this once. It will create AD_NC_new.
    split_val_set('your_original_data_path/AD_NC', 'new_store_data_path/AD_NC_new', 0.1)

    # set your data path
    train_dir = 'new_store_data_path/AD_NC_new/train'
    val_dir = 'new_store_data_path/AD_NC_new/val'
    test_dir = 'new_store_data_path/AD_NC_new/test'

    train_loader, val_loader, test_loader = get_data_loaders(train_dir, val_dir, test_dir)

    model = GFNet(img_size=224, patch_size=16, in_chans=1, num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    if os.path.exists('gfnet_model_latest.pth'):
        load_model(model, optimizer, 'gfnet_model_latest.pth')

    num_epochs = 50

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"Epoch {epoch + 1}/{num_epochs}. Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}. val Loss: {val_loss:.4f}, val Acc: {val_acc:.4f}. time: {run_time}")

    print("saving model parameters")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'gfnet_model_latest.pth')

    print("testing the model")
    final_validate(model, test_loader, criterion, device)

    print("finish model train, save and test")
