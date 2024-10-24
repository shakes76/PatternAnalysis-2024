"""
This file contains the code used to train, validate, test the GFNet model based
on the ADNI dataset. The best epoch with the best validation accuracy saves the
model weights to be used as inference. Performance from training is also 
plotted and saved as a figure.

Author: Kevin Gu
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from modules import GFNet
from dataset import load_adni_data
from functools import partial
import time
from timm.scheduler import create_scheduler
import matplotlib.pyplot as plt

from utils import get_dataset_root, get_device

def initialise_model(device: str):
    model = GFNet(
            patch_size=16, embed_dim=256, depth=12, mlp_ratio=4, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
    return model

def initialise_optimizer(model: nn.Module):
    optimizer = optim.Adam(model.parameters(), lr=0.00001, amsgrad=True)  # Adam optimizer with initial learning rate 0.01
    return optimizer


def initialise_scheduler(optimizer: torch.optim):
    # Cosine Annealing Learning Rate Scheduler with minimum learning rate (eta_min)
    # scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5, verbose=True)
    return scheduler

def plot_graphs(train_loss: list, validation_loss: list, train_accuracy: list, validation_accuracy: list):
    """
    Function to plot the model's loss and accuracy over the epochs during training

    Parameters:
        train_loss: A list of losses recorded during training
        train_accuracy: A list of accuracy values on based on train set 
                recorded during training
        validation_loss: A list of validation loss values recorded during 
                training
        validation_accuracy: A list of accuracy values on tested on validation 
                set recorded during training

    Returns: None
    """
    labels = ["Loss", "Accuracy"]
    fig, axs = plt.subplots(2, 1)
    train_data = [train_loss, train_accuracy]
    val_data = [validation_loss, validation_accuracy]

    for index, ax in enumerate(axs):
        ax.plot(train_data[index], label=f"Training {labels[index]}", color="blue")
        ax.plot(val_data[index], label=f"Validation {labels[index]}", color="red")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(labels[index])
        ax.set_title(f"Training and Validation {labels[index]}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("training_figures.png")


def test_model(model: nn.Module, device: str, root_dir: str) -> float:
    """
    Test the model on the test dataset.

    Parameters:
        model: Model to test on
        device: Device to run testing on
        root_dir: Root directory of ADNI dataset

    Returns:
        accuracy: Accuracy as a percentage from testing on test set
    """
     # Testing loop
    testing_start = time.time()
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    test_loader = load_adni_data(root_dir=root_dir, testing=True)
    with torch.no_grad():  # No need to compute gradients during evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Get predicted classes
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    testing_time = time.time() - testing_start
    print(f"Testing took {testing_time} seconds or {testing_time / 60} minutes")
    print("Accuracy", accuracy)

    return accuracy

def main():

    device = get_device()
    root_dir = get_dataset_root()

    train_loader, val_loader = load_adni_data(root_dir=root_dir)

    # Assuming you already have a model and dataloaders (train_loader, val_loader)
    model = initialise_model(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Use the appropriate loss function
    optimizer = initialise_optimizer(model)
    scheduler = initialise_scheduler(optimizer)

    # Training loop
    n_epochs = 100

    train_loss_list = []
    validation_loss_list = []
    train_accuracy_list = []
    validation_accuracy_list = [] 

    best_validation_accuracy = 0

    start_time = time.time()
    for epoch in range(n_epochs):
        # Training phase
        model.train()  # Set the model to training mode

        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            # Move data to the GPU if available
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # Get predicted classes
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()  # Count correct predictions

        # Calculate training loss and accuracy
        training_loss = running_loss / len(train_loader)
        training_accuracy = 100 * correct_train / total_train

        train_loss_list.append(training_loss)
        train_accuracy_list.append(training_accuracy)

            
        # Validation phase
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
        
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_val_loss += loss.item()  # Accumulate loss
                _, predicted = torch.max(outputs.data, 1)  # Get predicted classes
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()  # Count correct predictions

        validation_loss = running_val_loss / len(val_loader)
        validation_accuracy = 100 * correct_val / total_val

        validation_loss_list.append(validation_loss)
        validation_accuracy_list.append(validation_accuracy)

        # Step the scheduler
        scheduler.step(validation_loss)

        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{n_epochs}], '
              f'Train Loss: {training_loss:.4f}, Train Accuracy: {training_accuracy:.2f}%, '
              f'Val Loss: {validation_loss:.4f}, Val Accuracy: {validation_accuracy:.2f}%'
              f'Learning Rate: {optimizer.param_groups[0]["lr"]:.8f}')
        
        # Checkpoint based on best validation accuracy
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'validation_accuracy': validation_accuracy_list,
                'validation_loss': validation_loss_list,
                'training_accuracy': train_accuracy_list,
                'training_loss': train_loss_list
            }

            torch.save(checkpoint, 'model_checkpoint_actual.pth')

        
    training_time = time.time() - start_time
    print(f"Training took {training_time} seconds or {training_time / 60} minutes")



    # Testing loop
    accuracy = test_model(model, device, root_dir)

    # testing_start = time.time()
    # model.eval()  # Set the model to evaluation mode
    # correct = 0
    # total = 0
    # test_loader = load_adni_data(root_dir=root_dir, testing=True)
    # with torch.no_grad():  # No need to compute gradients during evaluation
    #     for inputs, labels in test_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs.data, 1)  # Get predicted classes
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    # accuracy = correct / total
    # testing_time = time.time() - testing_start
    # print(f"Testing took {testing_time} seconds or {testing_time / 60} minutes")
    # print("Accuracy", accuracy)

    # print("\n\n\n\n")
    # print("Validation accuracy:", validation_accuracy_list)
    # print("Validation loss:", validation_loss_list)
    # print("Training accuracy:", train_accuracy_list)
    # print("Training loss:", train_loss_list)




        
if __name__ == "__main__":
    main()
    
