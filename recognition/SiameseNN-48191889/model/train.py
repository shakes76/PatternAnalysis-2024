import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from dataset import getDataLoader
from modules import SiameseNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.0001
num_epochs = 150
batch_size = 32
patience = 5
min_delta = 0.001

# Early stopping variables
best_loss = float("inf")
es_count = 0

# Data transformations
data_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# Load train dataset
train_loader = getDataLoader(data_transforms, batch_size, method="mixed")

# Create new model or load previous model
model = SiameseNetwork().to(device)

model_path = "siamese_model_final.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Model loaded successfully.")

# Initialize criterion and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []


# Iterate through the specified number of epochs and specified batch sizes
def train():
    global best_loss, es_count
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        # Progress bar for visualization
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

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 0:
                progress_bar.set_postfix({"Batch Loss": loss.item()})

        # Calculate losses for single epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Perform early stopping if loss outnumbers patience and min delta values
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            es_count = 0
        else:
            es_count += 1
            print(f"No improvement for {es_count} epochs.")

        if es_count >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Save final model
    torch.save(model.state_dict(), "siamese_model_final.pth")


def plot_losses(save_path="loss_plot.png"):

    # Create loss plot
    plt.plot(train_losses, label="Training Loss")
    plt.title("Loss during training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Save loss plot as image
    plt.savefig(save_path)
    plt.clf()


# Train function
train()
plot_losses()
