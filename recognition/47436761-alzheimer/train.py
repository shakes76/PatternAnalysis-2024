from modules import AlzheimerModel
from dataset import create_data_loader
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
from torch import GradScaler, autocast
import os


from torch.optim.lr_scheduler import StepLR
import time
from sklearn.metrics import confusion_matrix, classification_report

# Parameters
NUM_EPOCHS = 5
START_EPOCH = 0 # 0 index
train_dir = 'dataset/AD_NC/train'
test_dir = 'dataset/AD_NC/test'
csv_file_path = 'training_epochs.csv'
batch_size = 32
learning_rate = 0.001

# Define hyperparameters and settings
in_channels = 1
patch_size = 16
embed_size = 768
img_size = 224
num_layers = 12
num_heads = 8
d_mlp = 2048
dropout_rate = 0.1
num_classes = 2
batch_size = 32
learning_rate = 1e-3

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # Kaiming (He) initialization for layers with ReLU/GELU activation
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

if __name__ == "__main__":
    train_loader, val_loader = create_data_loader(
        train_dir, 
        batch_size=batch_size, 
        train=True, 
        val=True, 
        val_split=0.2, 
    )

    model = AlzheimerModel(
        in_channels=in_channels,
        patch_size=patch_size,
        embed_size=embed_size,
        img_size=img_size,
        num_layers=num_layers,
        num_heads=num_heads,
        d_mlp=d_mlp,
        dropout_rate=dropout_rate,
        num_classes=num_classes
    )
    device = torch.device('cuda')
    cuda_available = torch.cuda.is_available()
    print(f"Is CUDA available? {cuda_available}")
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8, weight_decay=1e-5)


    # Scaler for mixed precision
    scaler = GradScaler('cuda')

    if START_EPOCH > 0:
        checkpoint_path = f'output/param/alzheimer_vit_epoch_{START_EPOCH}.pth'
        model.load_state_dict(torch.load(checkpoint_path))
        print(f'Loaded model from {checkpoint_path}')
    else:
        # Apply the weight initialization for first epoch
        model.apply(initialize_weights)

    


    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(START_EPOCH, NUM_EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # Forward pass with Mixed Precision Training
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Scales loss for mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # Step the scheduler at the end of each epoch
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        val_running_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            with autocast('cuda'):
                for images, labels in val_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(images)
                    loss = criterion(outputs, labels)  # Calculate validation loss
                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        val_loss = val_running_loss / len(val_loader)

        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
        print("Classification Report:")
        print(classification_report(all_labels, all_preds))

        accuracy = 0 #100 * correct / total
        avg_loss = running_loss / len(train_loader)

        log_message = f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n'
        accuracy_message = f'Test Accuracy after epoch {epoch+1}: {accuracy:.2f}%\n'

        end_time = time.time()
        time_taken = end_time - start_time
        time_msg = f"Time taken {time_taken:3f} seconds\n"

        print(log_message + accuracy_message + time_msg)

        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w', newline='') as file:
                file.write('epoch,train_loss,val_loss\n')
        
        with open(csv_file_path, 'a') as file:
            file.write(f'{epoch+1},{avg_loss:4f},{val_loss:.4f}\n')

        with open('training_log.txt', 'a') as log_file:
            log_file.write(log_message)
            log_file.write(accuracy_message)
            log_file.write(time_msg)
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
       

        # Save the model checkpoint after each epoch
        torch.save(model.state_dict(), f'output/param/alzheimer_vit_epoch_{epoch+1}.pth')
        print(f'Model saved: alzheimer_vit_epoch_{epoch+1}.pth')

    