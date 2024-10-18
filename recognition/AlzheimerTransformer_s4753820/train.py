from modules import ViT
from dataset import get_dataloaders
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to create the plots
def save_plots(train_losses, val_losses, train_accuracies, val_accuracies, save_dir="plots", exp_name="experiment"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # Ensure plots directory exists
    
    # Plotting loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/{exp_name}_loss_plot.png")  # Save the plot
    plt.close()  # Close the figure to free memory
    
    # Plotting accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/{exp_name}_accuracy_plot.png")  # Save the plot
    plt.close()

# Function to train for one epoch
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    batch_losses = []

    for (i, (images, labels)) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        # Move data to device (GPU or CPU)
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients so we don't update wrongly

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        running_loss += loss.detach().cpu().item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) # Each batch size, except for potentially the last batch if it's not divisible by batch size yknow.
        correct += (predicted == labels).sum().item()

        ## Should we print batch loss data hmm
        if (i + 1) % 100 == 0:  # Print every 100 batches
            print(f"Batch {i+1}, Avg Loss: {loss.item():.4f}, Correct Predictions: {correct} / {total}")
            batch_losses.append(loss.item())  # Append batch loss to list
        
    # Return average loss and accuracy
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy, batch_losses

# Function to evaluate on validation data
def evaluate(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode (i.e model knows not to expect gradient changes so it's faster).
    running_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for (i, (images, labels)) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            images = images.to(device) # torch.Size([32, 3, 224, 224])
            labels = labels.to(device) # torch.Size([32])

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            ## Should we print batch loss data hmm
            if (i + 1) % 100 == 0:  # Print every 10 batches
                print(f"Validation Batch {i+1}, Avg Loss: {loss.item():.4f}, Correct Predictions: {total_correct} / {total}")

    # Return average loss and accuracy
    avg_loss = running_loss / len(dataloader) # Divide by length of dataloder, since running loss is for each batch. Avg loss is AVERAGE BATCH LOSS (over one epoch)!!
    accuracy = 100 * total_correct / total
    return avg_loss, accuracy

# Function to save only the best model checkpoint
def save_model_checkpoint(model, optimizer, epoch, val_accuracy, path="models/best_model.pth", exp_name="test"):
    # Create the directory if it doesn't exist
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Always save to the same file to overwrite old models
    save_path = model_dir / f"{exp_name}best_model.pth"
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy
    }, save_path)
    
    print(f"Best model saved at epoch {epoch+1} with validation accuracy {val_accuracy:.2f}% at {save_path}")



# Function for full training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, exp_name):
    best_val_loss = float('inf')

    # Lists to store training/validation loss and accuracy for each epoch
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Optional but I like
    all_batch_losses = []  # List to store batch losses for all epochs

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train for one epoch
        train_loss, train_accuracy, batch_losses = train_one_epoch(model, train_loader, optimizer, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        all_batch_losses.extend(batch_losses)
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        # Evaluate on validation set each epoch
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Save the model if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model_checkpoint(model, optimizer, epoch, val_accuracy, exp_name=exp_name)

    print("Training complete")
    # Return losses and accuracies for plotting
    return train_losses, train_accuracies, val_losses, val_accuracies, all_batch_losses

def main():
    parser = argparse.ArgumentParser()

    # add arguments for hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--patch_size', type=int, default=16, help='Number of pixels a patch height/width has. E.g 16 gives 16x16 patch sizes.')
    parser.add_argument('--image_size', type=int, default=224, help='Height/Width size to set images to.')
    parser.add_argument('--path', type=str, default="/home/groups/comp3710/ADNI/AD_NC", help='Path to the dataset')

    parser.add_argument('--transformer_layers', type=int, default=12, help='Number of transformer layers you want the model to have. Default 12.')
    # Plotting flag (default True)
    parser.add_argument('--plot', dest='plot', action='store_true', default=True, help='Enable plotting (default is True)')
    parser.add_argument('--no-plot', dest='plot', action='store_false', help='Disable plotting')
    # Add experiment name to argument parser
    parser.add_argument('--exp_name', type=str, default="experiment", help='Name of the experiment for naming plots and models')

    # Parse arguments
    if torch.cuda.is_available():
        print("GPU available yay!!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Get device (hopefully gpu!!)

    args = parser.parse_args()

    lr  = args.lr
    num_epochs = args.num_epochs
    patch_size = args.patch_size
    batch_size = args.batch_size
    image_size = args.image_size 
    dataset_path = args.path
    num_transformer_layers = args.transformer_layers
    exp_name = args.exp_name

    # Get all the loaders 
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(batch_size=batch_size,
                                                                         image_size=image_size,
                                                                         path = dataset_path)
    
    # Create the model and optimisers 
    model = ViT(img_size=image_size, patch_size=patch_size, num_classes=2, num_transformer_layers=num_transformer_layers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ## Train the model
    train_losses, train_accuracies, val_losses, val_accuracies, all_batch_losses  = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs = num_epochs, device=device, exp_name = exp_name)

    print("stuff", train_losses, train_accuracies, val_losses, val_accuracies, all_batch_losses)

    ## Test model on test dataset for final evaluation
    test_loss, test_acc = evaluate(model, test_dataloader, criterion, device)
    print(f"Test: Loss {test_loss:.4f}, Accuracy {test_acc:.4f}")
    print("Model finished training!")


        # Save plots if required
    if args.plot:
        print("Plotting!")
        save_plots(train_losses, val_losses, train_accuracies, val_accuracies, save_dir="plots", exp_name=exp_name)


if __name__ == "__main__":
    main()