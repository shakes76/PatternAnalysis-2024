# train.py
import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from modules import SiameseNetwork
from torchvision import transforms
from dataset import getDataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.0001
num_epochs = 50
batch_size = 32
patience = 5  # Early stopping patience: stop if no improvement after 'patience' epochs
min_delta = 0.001  # Minimum change to qualify as an improvement

data_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# Initialize dataset
train_loader = getDataLoader(data_transforms, batch_size, method="mixed")

# Initialize model
model = SiameseNetwork().to(device)

# Load the saved model weights if they exist
model_path = "siamese_model_final.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path), strict=False)
    print("Model loaded successfully.")

# Loss and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store losses for plotting
train_losses = []

# Early stopping variables
best_loss = float("inf")
early_stop_counter = 0


def train():
    global best_loss, early_stop_counter
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
        )

        for i, (img1, img2, labels) in progress_bar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            # Forward pass
            outputs = model(img1, img2).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update progress bar with loss info every 10 batches
            if i % 10 == 0:
                progress_bar.set_postfix({"Batch Loss": loss.item()})

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Early stopping check
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            early_stop_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "siamese_model_best.pth")
            print("Best model saved.")
        else:
            early_stop_counter += 1
            print(f"No improvement for {early_stop_counter} epochs.")

        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Save final model
    torch.save(model.state_dict(), "siamese_model_final.pth")


def plot_losses(save_path="loss_plot.png"):
    plt.plot(train_losses, label="Training Loss")
    plt.title("Loss during training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(save_path)
    plt.clf()


# Call the train function to resume training
train()
plot_losses()
