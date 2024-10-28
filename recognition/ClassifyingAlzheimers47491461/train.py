from modules import *
from dataset import *
import torch.nn as nn
import torch.optim as optim
import torch.fft
import numpy as np

def train_model():
    # Instantiate GFNet model
    model = GFNet(
        img_size=224, patch_size=14, in_chans=1, num_classes=2, embed_dim=256, depth=12,
        mlp_ratio=4., drop_rate=0.1, drop_path_rate=0.1).to(device)
    print("model loaded")

    # DataLoader
    dataloader = process(colab=True)
    print("data processed")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # Learning rate scheduler
    num_epochs = 50
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

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss {avg_loss:.4f}")
        loss_list.append(avg_loss)
        epoch_list.append(epoch + 1)

    model_save_path = '/content/drive/MyDrive/model_state.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return loss_list, epoch_list

# Device configuration
device = torch.device("cuda")

if __name__ == "__main__":
    loss_list, epoch_list = train_model()
    print(f"Losses: {loss_list}")
    print(f"Epochs: {epoch_list}")