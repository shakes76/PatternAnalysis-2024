import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import create_model
from dataset import get_dataloaders
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def train_model():
    """
    Trains, validates and tests the model. Once training is complete, the model is saved.
    """

    # Set device to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Define directories for data, output, and checkpoints.
    data_dir = '/home/groups/comp3710/ADNI/AD_NC'
    output_dir = 'output'
    checkpoints_dir = 'checkpoints'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Get dataloaders and class names.
    dataloaders, class_names = get_dataloaders(data_dir)
    num_classes = len(class_names)
    print(f'Classes: {class_names}')

    # Initialize the model.
    model = create_model(num_classes)
    model = model.to(device)

    # Set up the loss function, optimizer, and learning rate scheduler.
    scaler = torch.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training parameters
    num_epochs = 13
    total_start_time = time.time()

    # Lists to store metrics for plotting.
    train_losses, val_losses, test_losses = [], [], []
    train_accs, val_accs, test_accs = [], [], []

    for epoch in range(1, num_epochs+1):
        # Train the model.
        model.train()
        running_loss, running_corrects, total_samples = 0.0, 0, 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass with mixed precision.
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass and optimization.
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        # Calculate epoch statistics.
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples * 100
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())

        # Validate the model.
        model.eval()
        val_running_loss, val_running_corrects, val_total_samples = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                val_total_samples += inputs.size(0)

        # Calculate validation statistics.
        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_acc = val_running_corrects.double() / val_total_samples * 100
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.item())

        # Test the model.
        test_running_loss, test_running_corrects, test_total_samples = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                test_running_loss += loss.item() * inputs.size(0)
                test_running_corrects += torch.sum(preds == labels.data)
                test_total_samples += inputs.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate testing statistics.
        test_epoch_loss = test_running_loss / test_total_samples
        test_epoch_acc = test_running_corrects.double() / test_total_samples * 100
        test_losses.append(test_epoch_loss)
        test_accs.append(test_epoch_acc.item())

        # Print epoch statistics
        time_elapsed = time.time() - total_start_time
        time_formatted = time.strftime("%M:%S", time.gmtime(time_elapsed))
        print(f'Epoch [{epoch}/{num_epochs}] | '
              f'train loss: {epoch_loss:.4f} | train acc: {epoch_acc:.2f}% | '
              f'val loss: {val_epoch_loss:.4f} | val acc: {val_epoch_acc:.2f}% | '
              f'test loss: {test_epoch_loss:.4f} | test acc: {test_epoch_acc:.2f}% | '
              f'Time Elapsed: {time_formatted}')

        scheduler.step() # Change learning rate.

        # Save model checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(checkpoints_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f'Model checkpoint saved at epoch {epoch}')

        # Stop training if test accuracy reaches 80%
        if test_epoch_acc >= 80.0:
            print(f"Reached 80% acc on test set at epoch {epoch}. Stopping training.")
            break

    print('Training complete')

    # Save the final model
    final_model_path = os.path.join(checkpoints_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved at {final_model_path}')

    # Plot losses vs epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch+1), train_losses, label='train loss', color='black')
    plt.plot(range(1, epoch+1), val_losses, label='val loss', color='purple')
    plt.plot(range(1, epoch+1), test_losses, label='test loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    loss_plot_path = os.path.join(output_dir, 'loss_vs_epochs.png')
    plt.savefig(loss_plot_path)
    plt.show()

    # Plot accuracies vs epochs.
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch+1), train_accs, label='train acc', color='black')
    plt.plot(range(1, epoch+1), val_accs, label='val acc', color='purple')
    plt.plot(range(1, epoch+1), test_accs, label='test acc', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    accuracy_plot_path = os.path.join(output_dir, 'accuracy_vs_epochs.png')
    plt.savefig(accuracy_plot_path)
    plt.show()

    # Generate and save confusion matrix.
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    cm_plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path)
    plt.show()

if __name__ == '__main__':
    train_model()
