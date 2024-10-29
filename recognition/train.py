"""
containing the source code for training, validating, testing and saving the ViT.

The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”.
the losses and metrics during training are plotted
"""
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import time
# from dataset import ADNIDataset, get_dataloader
# from modules import ViTClassifier

# def train_model(model, train_loader, val_loader, epochs, lr, device):
#     """
#     Train the Vision Transformer model for Alzheimer's classification.
#     """
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()

#     train_losses = []
#     val_losses = []
#     train_accs = []
#     val_accs = []
#     stopping_epoch = epochs
#     down_consec = 0

#     for epoch in range(epochs):
#         # Training loop
#         model.train()
#         train_loss = 0.0
#         train_correct = 0
#         train_total = 0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             train_total += labels.size(0)
#             train_correct += (predicted == labels).sum().item()
#         train_loss /= len(train_loader)
#         train_acc = train_correct / train_total
#         train_losses.append(train_loss)
#         train_accs.append(train_acc)

#         # Validation loop
#         model.eval()
#         val_loss = 0.0
#         val_correct = 0
#         val_total = 0
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#                 _, predicted = torch.max(outputs, 1)
#                 val_total += labels.size(0)
#                 val_correct += (predicted == labels).sum().item()
#         val_loss /= len(val_loader)
#         val_acc = val_correct / val_total
#         val_losses.append(val_loss)
#         val_accs.append(val_acc)

#         print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

#         # Early stopping check
#         if epoch > 0 and val_acc < val_accs[-2]:
#             down_consec += 1
#         else:
#             down_consec = 0
#         if down_consec >= 4:
#             stopping_epoch = epoch + 1
#             break

#     # Save the best model
#     torch.save(model, "adni_vit.pt")

#     # Plot training and validation metrics
#     plot_metric(stopping_epoch, 'loss', train_losses, val_losses)
#     plot_metric(stopping_epoch, 'accuracy', train_accs, val_accs)

#     return model

# def plot_metric(stopping_epoch: int, metric_type: str, train_data: list, val_data: list):
#     """
#     Helper function to plot a given metric
#     """
#     plt.figure()
#     plt.plot(range(1, stopping_epoch+1), train_data, label = f"Training {metric_type}")
#     plt.plot(range(1, stopping_epoch+1), val_data, label=f"Validation {metric_type}", color='orange')
#     plt.xlabel('Epoch')
#     plt.ylabel(metric_type)
#     plt.legend()
#     plt.title(f"Training {metric_type} vs validation {metric_type}")
#     plt.savefig(f"Training_vs_validation_{metric_type}_{int(time.time())}.png")

# def main():
#     """
#     Main execution function.
#     """
#     # Device configuration
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Initialise hyperparameters
#     BATCH_SIZE = 64
#     EPOCHS = 20
#     LR = 1e-4

#     # Initialise data loaders
#     train_loader, val_loader = get_dataloader(batch_size=BATCH_SIZE, train=True)
#     test_loader = get_dataloader(batch_size=BATCH_SIZE, train=False)

#     # Initialise model
#     model = ViTClassifier().to(device)
#     print("nya")
#     # Run training
#     trained_model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=device)

# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from dataset import ADNIDataset, get_dataloader
from modules import ViTClassifier
from tqdm import tqdm
import sys

def train_model(model, train_loader, val_loader, epochs, lr, device):
    """
    Train the Vision Transformer model for Alzheimer's classification with progress tracking.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    stopping_epoch = epochs
    down_consec = 0
    best_val_acc = 0

    print(f"\nStarting training on device: {device}")
    print(f"Total epochs: {epochs}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}\n")

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training loop
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]',
                         leave=False, file=sys.stdout)

        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                  'acc': f'{(train_correct/train_total)*100:.2f}%'})

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]',
                       leave=False, file=sys.stdout)

        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Update progress bar
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                    'acc': f'{(val_correct/val_total)*100:.2f}%'})

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f'\nEpoch [{epoch+1}/{epochs}] - {epoch_time:.1f}s')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'Saving best model with validation accuracy: {val_acc*100:.2f}%')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, "adni_vit_best.pt")

        # Early stopping check
        if epoch > 0 and val_acc < val_accs[-2]:
            down_consec += 1
            print(f'Validation accuracy decreased. Counter: {down_consec}/4')
        else:
            down_consec = 0
        if down_consec >= 4:
            print('\nEarly stopping triggered!')
            stopping_epoch = epoch + 1
            break

        print('-' * 60 + '\n')

    total_time = time.time() - start_time
    print(f'\nTraining completed in {total_time/60:.1f} minutes')
    print(f'Best validation accuracy: {best_val_acc*100:.2f}%')

    # Plot training and validation metrics
    plot_metric(stopping_epoch, 'loss', train_losses, val_losses)
    plot_metric(stopping_epoch, 'accuracy', train_accs, val_accs)

    return model

def plot_metric(stopping_epoch: int, metric_type: str, train_data: list, val_data: list):
    """
    Helper function to plot a given metric
    """
    plt.figure()
    plt.plot(range(1, stopping_epoch+1), train_data, label = f"Training {metric_type}")
    plt.plot(range(1, stopping_epoch+1), val_data, label=f"Validation {metric_type}", color='orange')
    plt.xlabel('Epoch')
    plt.ylabel(metric_type)
    plt.legend()
    plt.title(f"Training {metric_type} vs validation {metric_type}")
    plt.savefig(f"Training_vs_validation_{metric_type}_{int(time.time())}.png")

def main():
    """
    Main execution function.
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print system information
    print("\nSystem Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device being used: {device}")
    print(f"Number of available GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    # Initialise hyperparameters
    BATCH_SIZE = 32  # Reduced batch size to help with memory
    EPOCHS = 20
    LR = 1e-4

    print("\nHyperparameters:")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of epochs: {EPOCHS}")
    print(f"Learning rate: {LR}")

    # Initialise data loaders
    print("\nLoading data...")
    train_loader, val_loader = get_dataloader(batch_size=BATCH_SIZE, train=True)
    test_loader = get_dataloader(batch_size=BATCH_SIZE, train=False)
    print("Data loaded successfully!")

    # Initialise model
    print("\nInitializing model...")
    model = ViTClassifier().to(device)
    print("Model initialized successfully!")

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Run training
    train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=device)

if __name__ == "__main__":
    main()