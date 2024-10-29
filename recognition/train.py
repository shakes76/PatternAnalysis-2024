"""
containing the source code for training, validating, testing and saving the ViT.

The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”.
the losses and metrics during training are plotted
"""
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from contextlib import nullcontext
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import time
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from dataset import ADNIDataset, get_dataloader, get_dataset
from modules import ViTClassifier

class OptimizedTrainer:
    def __init__(self, model, train_loader, val_loader, device,
                 mixed_precision=True, distributed=False, num_workers=4,
                 save_dir='checkpoints'):
        """
        Initialize optimized trainer with monitoring capabilities
        """
        self.model = model
        self.device = device
        self.mixed_precision = mixed_precision
        self.distributed = distributed
        self.num_workers = num_workers
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Setup mixed precision
        self.scaler = amp.GradScaler() if mixed_precision else None

        # Optimize data loading
        self.train_loader = self._optimize_dataloader(train_loader)
        self.val_loader = self._optimize_dataloader(val_loader)

        # Initialize distributed training if requested
        if distributed:
            self.model = DDP(model)

        # Initialize training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rate': [], 'epochs': []
        }
        self.best_val_acc = 0
        self.best_model_path = None

    def _optimize_dataloader(self, dataloader):
        """
        Optimize dataloader for speed
        """
        return DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=dataloader.dataset.train if hasattr(dataloader.dataset, 'train') else True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

    def train_epoch(self, optimizer, criterion, scheduler=None):
        """
        Run optimized training epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        amp_context = amp.autocast() if self.mixed_precision else nullcontext()
        pbar = tqdm(self.train_loader, desc='Training', leave=False)

        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad(set_to_none=True)

            with amp_context:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                            'acc': f'{(correct/total)*100:.2f}%'})

            del outputs, loss
            torch.cuda.empty_cache()

        return running_loss / len(self.train_loader), correct / total

    def validate_epoch(self, criterion):
        """
        Run validation epoch
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                'acc': f'{(correct/total)*100:.2f}%'})

        return running_loss / len(self.val_loader), correct / total

    def save_checkpoint(self, optimizer, epoch, train_loss, val_loss,
                       train_acc, val_acc, lr, is_best=False):
        """
        Save model checkpoint
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
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

        return checkpoint_path

    def plot_training_history(self):
        """
        Plot training metrics
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        ax1.plot(self.history['train_loss'], label='Training Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Training Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Accuracy Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()
        return fig

    def evaluate(self, test_loader, classes):
        """
        Evaluate model on test set
        """
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc='Evaluating'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        return y_true, y_pred, report

def train_model_optimized(model, train_dataset, val_dataset, test_dataset=None,
                         epochs=20, batch_size=32, lr=1e-4, classes=None,
                         early_stopping_patience=4):
    """
    Main training function with optimizations and monitoring
    """
    # Setup device and optimization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    num_workers = min(mp.cpu_count(), 8)

    # Optimize batch size based on GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        batch_size = min(batch_size, int(gpu_memory / (1024**3) * 4))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 10,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Initialize trainer
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        mixed_precision=True,
        distributed=(torch.cuda.device_count() > 1)
    )

    # Training loop
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    stopping_epoch = epochs
    down_consec = 0

    print(f"\nStarting training on device: {device}")
    print(f"Batch size: {batch_size}, Learning rate: {lr}")

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training and validation
        train_loss, train_acc = trainer.train_epoch(optimizer, criterion, scheduler)
        val_loss, val_acc = trainer.validate_epoch(criterion)

        # Check if best model
        is_best = val_acc > trainer.best_val_acc

        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint(
            optimizer, epoch, train_loss, val_loss,
            train_acc, val_acc, lr, is_best
        )

        # Print epoch results
        epoch_time = time.time() - epoch_start
        print(f'\nEpoch [{epoch+1}/{epochs}] - {epoch_time:.1f}s')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%')

        # Early stopping check
        if epoch > 0 and val_acc < trainer.history['val_acc'][-2]:
            down_consec += 1
            if down_consec >= early_stopping_patience:
                print('\nEarly stopping triggered!')
                stopping_epoch = epoch + 1
                break
        else:
            down_consec = 0

    # Final evaluation
    if test_dataset is not None and classes is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        y_true, y_pred, report = trainer.evaluate(test_loader, classes)

    # Plot and save training curves
    trainer.plot_training_history()
    plt.savefig(os.path.join(trainer.save_dir, 'training_curves.png'))

    total_time = time.time() - start_time
    print(f'\nTraining completed in {total_time/60:.1f} minutes')
    print(f'Best validation accuracy: {trainer.best_val_acc*100:.2f}%')

    return model, trainer.history

def main():
    """Main execution function"""
    # Device configuration and system information
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nSystem Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device being used: {device}")
    print(f"Number of available GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_props.name} ({gpu_props.total_memory / 1024**3:.1f} GB)")

    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-4
    CLASSES = ['CN', 'MCI', 'AD', 'SMC']
    EARLY_STOPPING_PATIENCE = 4

    print("\nHyperparameters:")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of epochs: {EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")

    # Initialize data
    print("\nLoading data...")
    # Get datasets instead of dataloaders since our framework will create optimized loaders
    train_dataset, val_dataset = get_dataset(train=True)  # Assuming you modify get_dataloader to return datasets
    test_dataset = get_dataset(train=False)
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

    # Additional optimization information
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("\nOptimization Settings:")
        print("cuDNN benchmark mode: Enabled")
        print(f"Mixed precision training: Enabled")
        print(f"Distributed training: {'Enabled' if torch.cuda.device_count() > 1 else 'Disabled'}")

    # Train model using optimized framework
    try:
        model, history = train_model_optimized(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            classes=CLASSES,
            early_stopping_patience=EARLY_STOPPING_PATIENCE
        )

        print("\nTraining completed successfully!")

        # Print final metrics
        final_train_acc = history['train_acc'][-1]
        final_val_acc = history['val_acc'][-1]
        best_val_acc = max(history['val_acc'])
        best_epoch = history['val_acc'].index(best_val_acc) + 1

        print("\nFinal Results:")
        print(f"Best validation accuracy: {best_val_acc*100:.2f}% (Epoch {best_epoch})")
        print(f"Final training accuracy: {final_train_acc*100:.2f}%")
        print(f"Final validation accuracy: {final_val_acc*100:.2f}%")

        # Save complete training results
        results = {
            'hyperparameters': {
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'learning_rate': LR,
                'early_stopping_patience': EARLY_STOPPING_PATIENCE
            },
            'model_info': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            },
            'training_history': history,
            'final_metrics': {
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
                'final_train_acc': final_train_acc,
                'final_val_acc': final_val_acc
            }
        }

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'training_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=4)

        print("\nResults have been saved. Check the output directory for:")
        print("- Model checkpoints")
        print("- Training curves")
        print("- Confusion matrix")
        print("- Complete training results JSON")

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

    return model, history

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise


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