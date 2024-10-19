import torch
import torch.optim as optim
import torch.nn.functional as F
from dataset import create_dataloaders  # Import dataset and loader
from modules import UNet  # Import the UNet model
import time
import matplotlib.pyplot as plt

# Using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 0.001
num_epochs = 4
batch_size = 64
num_classes = 6  # As there are 6 segmentation classes

# Paths to your data (set these paths based on your project structure)
train_images_folder = 0 #abscent for now
train_masks_folder = 0 #abscent for now
test_images_folder = 0 #abscent for now
test_masks_folder = 0 #abscent for now

# Load the datasets
train_loader = create_dataloaders(train_images_folder, train_masks_folder, batch_size, normImage=True)
val_loader = create_dataloaders(test_images_folder, test_masks_folder, batch_size, normImage=True)

# Initialize the model, loss function, and optimizer
net = UNet(num_classes=num_classes).to(device)  # Move the model to the device
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Dice Loss Function for multi-class segmentation
def dice_loss(pred, target, smooth=1):
    # Apply softmax to get probabilities for each class
    pred = F.softmax(pred, dim=1)
    
    # Ensure target is properly shaped as [batch_size, H, W]
    target = target.squeeze(dim=1) if target.dim() == 4 else target
    
    # Convert target to one-hot encoding with the correct shape
    target_flat = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).contiguous().view(pred.shape[0], pred.shape[1], -1)  # [batch_size, num_classes, H*W]
    
    # Flatten pred tensor to match the target
    pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)  # [batch_size, num_classes, H*W]
    
    # Calculate intersection and Dice score per class
    intersection = (pred_flat * target_flat).sum(dim=2)  # Sum over H*W (spatial dimensions)
    dice_score = (2. * intersection + smooth) / (pred_flat.sum(dim=2) + target_flat.sum(dim=2) + smooth)
    
    # Return the mean Dice loss across all classes
    return 1 - dice_score.mean()

# Training loop
def train_model():
    train_losses = []
    val_losses = []

    start_time = time.time()
    print("Starting training\n")

    for epoch in range(num_epochs):
        net.train()  # Set the model to training mode
        running_loss = 0.0

        # Iterate over the batches
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(images)

            # Calculate loss (using dice loss)
            loss = dice_loss(outputs, masks)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Print statistics every 2 batches
            if (i + 1) % 2 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 2:.3f}')
                running_loss = 0.0  # Reset running loss after every print

        # Store average training loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation step
        val_loss = validate_model()
        val_losses.append(val_loss)

    end_time = time.time()
    print('\nFinished Training')
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    # Save the model
    torch.save(net.state_dict(), "unet_model.pth")

    # Plot the training and validation losses
    plot_losses(train_losses, val_losses)

# Validation loop
def validate_model():
    net.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():  # No need to calculate gradients during validation
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = net(images)

            # Calculate loss
            print(f'Outputs shape: {outputs.shape}, Masks shape: {masks.shape}')
            loss = dice_loss(outputs, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.3f}')
    return avg_val_loss

# Function to plot the training and validation losses
def plot_losses(train_losses, val_losses):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Start training
if __name__ == "__main__":
    train_model()