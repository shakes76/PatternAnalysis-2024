"""
containing the source code for training, validating, testing and saving the ViT.

The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”.
the losses and metrics during training are plotted
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os
import time
from datetime import datetime
from tqdm import tqdm
import sys
from sklearn.metrics import confusion_matrix, classification_report

from dataset import ADNIDataset, get_dataloader
from modules import ViTClassifier

class ModelCheckpointing:
    def __init__(self, save_dir='checkpoints'):
        """Initialize checkpoint manager"""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.best_val_acc = 0
        self.best_model_path = None
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rate': [], 'epochs': []
        }

    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_loss,
                       train_acc, val_acc, lr, is_best=False):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'learning_rate': lr
        }

        # Update history
        for key in ['train_loss', 'val_loss', 'train_acc', 'val_acc']:
            self.history[key].append(eval(key))
        self.history['learning_rate'].append(lr)
        self.history['epochs'].append(epoch)

        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pt')
        torch.save(checkpoint, checkpoint_path)

        # Save best model if applicable
        if is_best:
            self.best_val_acc = val_acc
            self.best_model_path = os.path.join(self.save_dir, f'best_model_{timestamp}.pt')
            torch.save(checkpoint, self.best_model_path)

            # Save config
            config = {
                'timestamp': timestamp,
                'epoch': epoch,
                'best_val_acc': val_acc,
                'final_train_loss': train_loss,
                'final_val_loss': val_loss,
                'learning_rate': lr
            }

            with open(os.path.join(self.save_dir, f'model_config_{timestamp}.json'), 'w') as f:
                json.dump(config, f, indent=4)

        return checkpoint_path

    def load_checkpoint(self, model, optimizer, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, checkpoint

    def load_best_model(self, model, optimizer):
        """Load the best model"""
        if self.best_model_path is None:
            raise ValueError("No best model checkpoint found")
        return self.load_checkpoint(model, optimizer, self.best_model_path)

    def save_training_history(self):
        """Save training history"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_path = os.path.join(self.save_dir, f'training_history_{timestamp}.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        return history_path

class VisualizationUtils:
    @staticmethod
    def plot_training_history(history):
        """Plot training metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        ax1.plot(history['train_loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot accuracy
        ax2.plot(history['train_acc'], label='Training Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Accuracy Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_attention_maps(model, image, pred_class, true_class=None):
        """Plot attention maps"""
        with torch.no_grad():
            _ = model(image.unsqueeze(0))
        attention = model.get_attention_weights()
        attention = attention.mean(1).squeeze()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.imshow(image.permute(1, 2, 0).cpu())
        ax1.axis('off')
        ax1.set_title('Original Image')

        sns.heatmap(attention.cpu(), ax=ax2, cmap='viridis')
        title = f'Attention Map (Pred: {pred_class})'
        if true_class is not None:
            title += f' (True: {true_class})'
        ax2.set_title(title)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        return plt.gcf()

    @staticmethod
    def generate_classification_report(y_true, y_pred, classes):
        """Generate classification report"""
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        plt.figure(figsize=(10, 6))
        sns.heatmap(df_report.iloc[:-3, :-1].astype(float), annot=True, cmap='Blues', fmt='.2f')
        plt.title('Classification Report')
        return plt.gcf()

def train_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                         'acc': f'{(correct/total)*100:.2f}%'})

    return running_loss / len(loader), correct / total

def validate_epoch(model, loader, criterion, device):
    """Run one validation epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                             'acc': f'{(correct/total)*100:.2f}%'})

    return running_loss / len(loader), correct / total

def train_model(model, train_loader, val_loader, epochs, lr, device, early_stopping_patience=4):
    """Main training function"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    checkpointer = ModelCheckpointing()
    visualizer = VisualizationUtils()

    print(f"\nStarting training on device: {device}")
    print(f"Checkpoints will be saved to: {checkpointer.save_dir}")

    start_time = time.time()
    stopping_epoch = epochs
    down_consec = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Check if best model
        is_best = val_acc > checkpointer.best_val_acc

        # Save checkpoint
        checkpoint_path = checkpointer.save_checkpoint(
            model, optimizer, epoch, train_loss, val_loss,
            train_acc, val_acc, lr, is_best
        )

        epoch_time = time.time() - epoch_start
        print(f'\nEpoch [{epoch+1}/{epochs}] - {epoch_time:.1f}s')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%')
        print(f'Checkpoint saved: {checkpoint_path}')

        if is_best:
            print(f'New best model saved with validation accuracy: {val_acc*100:.2f}%')

        # Early stopping check
        if epoch > 0 and val_acc < checkpointer.history['val_acc'][-2]:
            down_consec += 1
            print(f'Validation accuracy decreased. Counter: {down_consec}/{early_stopping_patience}')
        else:
            down_consec = 0

        if down_consec >= early_stopping_patience:
            print('\nEarly stopping triggered!')
            stopping_epoch = epoch + 1
            break

        print('-' * 60)

    total_time = time.time() - start_time
    print(f'\nTraining completed in {total_time/60:.1f} minutes')
    print(f'Best validation accuracy: {checkpointer.best_val_acc*100:.2f}%')

    # Save final history
    history_path = checkpointer.save_training_history()
    print(f'Training history saved: {history_path}')

    # Plot training curves
    visualizer.plot_training_history(checkpointer.history)
    plt.savefig(os.path.join(checkpointer.save_dir, 'training_curves.png'))

    return model, checkpointer.history

def evaluate_model(model, test_loader, device, classes):
    """Evaluate model on test set"""
    model.eval()
    y_true = []
    y_pred = []
    visualizer = VisualizationUtils()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Generate evaluation plots
    cm_fig = visualizer.plot_confusion_matrix(y_true, y_pred, classes)
    cm_fig.savefig('confusion_matrix.png')

    report_fig = visualizer.generate_classification_report(y_true, y_pred, classes)
    report_fig.savefig('classification_report.png')

    return y_true, y_pred

def main():
    """Main execution function"""
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print system information
    print("\nSystem Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device being used: {device}")
    print(f"Number of available GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-4
    CLASSES = ['CN', 'MCI', 'AD', 'SMC']

    print("\nHyperparameters:")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of epochs: {EPOCHS}")
    print(f"Learning rate: {LR}")

    # Initialize data loaders
    print("\nLoading data...")
    train_loader, val_loader = get_dataloader(batch_size=BATCH_SIZE, train=True)
    test_loader = get_dataloader(batch_size=BATCH_SIZE, train=False)
    print("Data loaded successfully!")

    # Initialize model
    print("\nInitializing model...")
    model = ViTClassifier().to(device)
    print("Model initialized successfully!")

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        device=device
    )

    # Evaluate model
    y_true, y_pred = evaluate_model(model, test_loader, device, CLASSES)

    print("\nTraining and evaluation completed successfully!")
    print("Check the output directory for visualization plots and model checkpoints.")

if __name__ == "__main__":
    main()

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import time
# from dataset import ADNIDataset, get_dataloader
# from modules import ViTClassifier
# from tqdm import tqdm
# import sys

# def train_model(model, train_loader, val_loader, epochs, lr, device):
#     """
#     Train the Vision Transformer model for Alzheimer's classification with progress tracking.
#     """
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()

#     train_losses = []
#     val_losses = []
#     train_accs = []
#     val_accs = []
#     stopping_epoch = epochs
#     down_consec = 0
#     best_val_acc = 0

#     print(f"\nStarting training on device: {device}")
#     print(f"Total epochs: {epochs}")
#     print(f"Training batches per epoch: {len(train_loader)}")
#     print(f"Validation batches per epoch: {len(val_loader)}\n")

#     start_time = time.time()

#     for epoch in range(epochs):
#         epoch_start = time.time()

#         # Training loop
#         model.train()
#         train_loss = 0.0
#         train_correct = 0
#         train_total = 0

#         # Progress bar for training
#         train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]',
#                          leave=False, file=sys.stdout)

#         for images, labels in train_pbar:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             train_total += labels.size(0)
#             train_correct += (predicted == labels).sum().item()

#             # Update progress bar
#             train_pbar.set_postfix({'loss': f'{loss.item():.4f}',
#                                   'acc': f'{(train_correct/train_total)*100:.2f}%'})

#         train_loss /= len(train_loader)
#         train_acc = train_correct / train_total
#         train_losses.append(train_loss)
#         train_accs.append(train_acc)

#         # Validation loop
#         model.eval()
#         val_loss = 0.0
#         val_correct = 0
#         val_total = 0

#         # Progress bar for validation
#         val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]',
#                        leave=False, file=sys.stdout)

#         with torch.no_grad():
#             for images, labels in val_pbar:
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 val_total += labels.size(0)
#                 val_correct += (predicted == labels).sum().item()

#                 # Update progress bar
#                 val_pbar.set_postfix({'loss': f'{loss.item():.4f}',
#                                     'acc': f'{(val_correct/val_total)*100:.2f}%'})

#         val_loss /= len(val_loader)
#         val_acc = val_correct / val_total
#         val_losses.append(val_loss)
#         val_accs.append(val_acc)

#         epoch_time = time.time() - epoch_start

#         # Print epoch summary
#         print(f'\nEpoch [{epoch+1}/{epochs}] - {epoch_time:.1f}s')
#         print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
#         print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%')

#         # Save best model
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             print(f'Saving best model with validation accuracy: {val_acc*100:.2f}%')
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'train_loss': train_loss,
#                 'val_loss': val_loss,
#                 'train_acc': train_acc,
#                 'val_acc': val_acc,
#             }, "adni_vit_best.pt")

#         # Early stopping check
#         if epoch > 0 and val_acc < val_accs[-2]:
#             down_consec += 1
#             print(f'Validation accuracy decreased. Counter: {down_consec}/4')
#         else:
#             down_consec = 0
#         if down_consec >= 4:
#             print('\nEarly stopping triggered!')
#             stopping_epoch = epoch + 1
#             break

#         print('-' * 60 + '\n')

#     total_time = time.time() - start_time
#     print(f'\nTraining completed in {total_time/60:.1f} minutes')
#     print(f'Best validation accuracy: {best_val_acc*100:.2f}%')

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

#     # Print system information
#     print("\nSystem Information:")
#     print(f"PyTorch version: {torch.__version__}")
#     print(f"Device being used: {device}")
#     print(f"Number of available GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

#     # Initialise hyperparameters
#     BATCH_SIZE = 32  # Reduced batch size to help with memory
#     EPOCHS = 20
#     LR = 1e-4

#     print("\nHyperparameters:")
#     print(f"Batch size: {BATCH_SIZE}")
#     print(f"Number of epochs: {EPOCHS}")
#     print(f"Learning rate: {LR}")

#     # Initialise data loaders
#     print("\nLoading data...")
#     train_loader, val_loader = get_dataloader(batch_size=BATCH_SIZE, train=True)
#     test_loader = get_dataloader(batch_size=BATCH_SIZE, train=False)
#     print("Data loaded successfully!")

#     # Initialise model
#     print("\nInitializing model...")
#     model = ViTClassifier().to(device)
#     print("Model initialized successfully!")

#     # Print model summary
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"\nModel Parameters:")
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")

#     # Run training
#     train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=device)

# if __name__ == "__main__":
#     main()