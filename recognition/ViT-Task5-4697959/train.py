# train.py

import os
import copy
import time  
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from modules import VisionTransformer  
from dataset import get_data_loaders 
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import random
import itertools


def set_seed(seed=42):
    """
    Set the seed for reproducibility.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    
    # For CUDA algorithms, ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, patience=5, save_dir='saved_models'):
    """
    Trains the Vision Transformer model.

    Args:
        model (nn.Module): The Vision Transformer model.
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoaders.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to train on.
        num_epochs (int): Number of training epochs.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        save_dir (str): Directory to save the best model and plots.

    Returns:
        nn.Module: The trained model with best validation accuracy.
        dict: Training and validation loss history.
        dict: Training and validation accuracy history.
    """

    os.makedirs(save_dir, exist_ok=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    counter = 0

    # main training/validation loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch +1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track gradients only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only in train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Compute epoch loss and accuracy
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(save_dir, 'best_vit_model.pth'))
                    print("Best model updated and saved.\n")
                    counter = 0
                else:
                    counter +=1
                    print(f"No improvement for {counter} epochs.\n")
                    if counter >= patience:
                        print("Early stopping triggered.")
                        model.load_state_dict(best_model_wts)
                        return model, history

    print(f'Training complete. Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_vit_model.pth'))
    print(f"Final model saved at {os.path.join(save_dir, 'final_vit_model.pth')}")

    return model, history

def plot_metrics(history, save_dir='saved_models'):
    """
    Plots training and validation loss and accuracy.

    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_dir (str): Directory to save the plots.
    """
    epochs = range(1, len(history['train_loss']) +1)

    # Plot Loss
    plt.figure(figsize=(10,5))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10,5))
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy_curve.png'))
    plt.close()

def evaluate_model(model, dataloader, device, class_names, save_dir='saved_models'):
    """
    Evaluates the model on the test set and prints classification metrics.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Test DataLoader.
        device (torch.device): Device to perform evaluation on.
        class_names (list): List of class names.
        save_dir (str): Directory to save the confusion matrix plot.
    """
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0  # Initialize correct predictions counter
    total = 0    # Initialize total predictions counter

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate and print test accuracy
    test_accuracy = correct.double() / total
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:\n", report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_normalized[i, j]:.2f})",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()


def main():

    # Set the seed for reproducibility
    set_seed(42)

    # Start the timer
    start_time = time.time()

    # Configuration
    data_dir = "/home/groups/comp3710/ADNI/AD_NC"  # Dataset path
    batch_size = 32 # change the batch size 
    img_size = 224
    val_split = 0.2
    num_workers = 4
    num_classes = 2
    emb_size = 768
    num_heads = 12
    depth = 12
    ff_dim = 3072
    dropout = 0.1 # increased dropout
    patch_size = 16
    cls_token = True
    num_epochs = 80 
    patience = 50 # num of epochs before early stopping 
    learning_rate = 3e-4 # increased learning rate
    weight_decay = 1e-5 # decreased weight decay
    save_dir = 'saved_models'
    
    print(f"Key parameters for this trainning session: Batch size: {batch_size}, Dropout: {dropout}, Learning Rate: {learning_rate}, Epoch: {num_epochs}, Weight Decay: {weight_decay}")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders, class names, and class weights
    dataloaders, class_names, class_weights = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size,
        val_split=val_split,
        num_workers=num_workers
    )

    # Initialize the Vision Transformer model
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        emb_size=emb_size,
        num_heads=num_heads,
        depth=depth,
        ff_dim=ff_dim,
        num_classes=num_classes,
        dropout=dropout,
        cls_token=cls_token
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train the model
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        save_dir=save_dir
    )

    # Plot training metrics
    plot_metrics(history, save_dir=save_dir)
    print(f"Training and validation metrics plotted and saved to {save_dir}.")

    # Evaluate the model on the test set
    evaluate_model(
        model=model,
        dataloader=dataloaders['test'],
        device=device,
        class_names=class_names,
        save_dir=save_dir
    )
    print(f"Evaluation on test set completed. Confusion matrix saved to {save_dir}.")

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total time taken: {minutes}m {seconds}s")

if __name__ == '__main__':
    main()