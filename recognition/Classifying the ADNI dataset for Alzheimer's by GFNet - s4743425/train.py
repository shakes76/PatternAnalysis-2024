"""
This file contains the source code for the training, validating and testing of the model.
the model itself from modules.py is imported and is trained with the data from dataset.py
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
from modules import *
from dataset import *
from sklearn import metrics
from timm.scheduler import create_scheduler
from types import SimpleNamespace

# Set Hyperparameters
num_epochs = 300

learning_rate = 0.0005

batch_size = 32

def initialise():
    # Set up thee device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set directory to save plots
    asset_dir = 'assets'
    if not os.path.exists(asset_dir):
        os.makedirs(asset_dir)
    return device, asset_dir

# train the model
### inlcude 
def train(device, asset_dir, model, criterion, optimizer, scheduler, train_loader, val_loader):
    print("Start Training ...")
     # Start timer for training
    start_time = time.time()
    
    model.train()
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validate the model at the end of each epoch
        val_loss, val_acc = validate(device, model, criterion, val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Step the scheduler at the end of each epoch and update
        scheduler.step(epoch)
        
        # Print loss and accuracy for each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    # End timer for training
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in: {training_time:.2f} seconds")

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(asset_dir, 'training_and_validation_losses.png'))
    output_image_path = os.path.join(asset_dir, 'training_and_validation_losses.png')
    plt.savefig(output_image_path)
    plt.show()
    plt.close()

    # Plot validation accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    output_image_path = os.path.join(asset_dir, 'validation_accuracies.png')
    plt.savefig(output_image_path)
    plt.show()
    plt.close()

# for validating
def validate(device, model, criterion, val_loader):
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Validation loss
            val_loss += loss.item()

            # Calculate validation accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct_predictions / total_predictions

    return val_loss, val_accuracy

# Test function
def test(device, asset_dir, model, criterion, test_loader):
    print("Start Testing ...")
    
    # Start timer for testing
    start_time = time.time()

    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate test accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            # Save all predictions and labels for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = correct_predictions / total_predictions

    # End timer for testing
    end_time = time.time()
    testing_time = end_time - start_time
    print(f"Testing completed in: {testing_time:.2f} seconds")
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    confusion_matrix = metrics.confusion_matrix(all_labels, all_predictions)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)

    # Plot confusion matrix and save it to the 'assets' directory
    plt.figure(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    output_image_path = os.path.join(asset_dir, 'confusion_matrix.png')
    plt.savefig(output_image_path)
    plt.show()
    plt.close

    return test_loss, test_accuracy

# Main function to start training
def main():
    #intitialse the startup
    device, asset_dir = initialise()
    # Load datasets
    train_loader, val_loader = train_dataloader(batch_size=batch_size)
    test_loader, _ = test_dataloader(batch_size=batch_size)

    #load model
    model = GFNet(
        img_size=256,
        patch_size= 16,
        embed_dim=512,
        num_classes=2,
        in_channels=1,
        drop_rate=0.5,
        depth=19,
        mlp_ratio=4.,
        drop_path_rate=0.25,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Ceate a learning rate scheduler
    args = SimpleNamespace()
    args.sched = 'cosine'
    args.num_epochs=num_epochs
    args.decay_epochs=30
    args.min_lr=1e-6
    args.warmup_lr=1e-5
    args.cooldown_epochs=10

    scheduler, _ = create_scheduler(args, optimizer)

    #train the model and produce training results
    train(device, asset_dir, model, criterion, optimizer, scheduler, train_loader, val_loader)

    # Save the model after training
    model_save_path = "trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
