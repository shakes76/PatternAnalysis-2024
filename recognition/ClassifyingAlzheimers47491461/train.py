import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib as plt
from modules import *
from dataset import *


def train_model(colab=False):
    device = torch.device("cuda")
    # instantiate gfnet model
    model = GFNet(
        img_size=224, patch_size=16, num_classes=2, embed_dim=768, depth=12).to(device)
    print("Model loaded.")

    dataloader = process(colab=colab)
    print("Data processed.")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    num_epochs = 20
    warmup_epochs = 5
    total_steps = num_epochs * len(dataloader)
    warmup_steps = warmup_epochs * len(dataloader)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, 0.5 * (1.0 + np.cos(np.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    model.train()
    global_step = 0
    loss_list = []
    epoch_list = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        loss_list.append(avg_loss)
        epoch_list.append(epoch + 1)

    if colab:
        torch.save(model.state_dict(), '/content/drive/MyDrive/model_state.pth')
        print("path saved")

    return loss_list, epoch_list


if __name__ == "__main__":
    loss, epoch = train_model(colab=False)
