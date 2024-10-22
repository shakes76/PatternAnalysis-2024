import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from modules import GFNet
from dataset import load_adni_data
import platform
from functools import partial
import time
from timm.scheduler import create_scheduler

def main():

    if platform.system() == "Windows":
        root_dir = 'ADNI_AD_NC_2D/AD_NC'
    else:
        root_dir = '/home/groups/comp3710/ADNI/AD_NC'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = load_adni_data(root_dir=root_dir)

    # Assuming you already have a model and dataloaders (train_loader, val_loader)
    model = GFNet(
            patch_size=16, embed_dim=512, depth=19, mlp_ratio=4, drop_path_rate=0.25,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Use the appropriate loss function
    optimizer = optim.Adam(model.parameters())  # Adam optimizer with initial learning rate 0.01

    # Training loop
    n_epochs = 200

    # Cosine Annealing Learning Rate Scheduler with minimum learning rate (eta_min)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    train_loss_list = []
    validation_loss_list = []
    train_accuracy_list = []
    validation_accuracy_list = [] 


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

        # Step the scheduler based on validation loss
        scheduler.step(validation_loss)

        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{n_epochs}], '
              f'Train Loss: {training_loss:.4f}, Train Accuracy: {training_accuracy:.2f}%, '
              f'Val Loss: {validation_loss:.4f}, Val Accuracy: {validation_accuracy:.2f}%'
              f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
    training_time = time.time() - start_time
    print(f"Training took {training_time} seconds or {training_time / 60} minutes")



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

    print("\n\n\n\n")
    print("Validation accuracy:", validation_accuracy_list)
    print("Validation loss:", validation_loss_list)
    print("Training accuracy:", train_accuracy_list)
    print("Training loss:", train_loss_list)
    torch.save(model.state_dict(), "trained_model_weights.pth")


        
if __name__ == "__main__":
    main()
    
