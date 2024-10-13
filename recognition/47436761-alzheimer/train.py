from modules import AlzheimerModel
from dataset import *
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
NUM_EPOCHS = 40
START_EPOCH = 30 # 0 index
train_dir = '/home/groups/comp3710/ADNI/AD_NC/train' # '/home/groups/comp3710/ADNI/AD_NC/train'
test_dir = '/home/groups/comp3710/ADNI/AD_NC/test' # '/home/groups/comp3710/ADNI/AD_NC/train'
csv_file_path = 'training_epochs.csv'
batch_size = 32
learning_rate = 0.001

# Define hyperparameters and settings with minimal values
in_channels = 1
img_size = 224
patch_size = 16
embed_size = 768
num_layers = 12
num_heads = 8
d_mlp = 2048
dropout_rate = 0.4
num_classes = 2
batch_size = 32
learning_rate = 1e-5
weight_decay = 1e-4


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # Xavier (Glorot) initialization for linear layers
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    elif isinstance(m, nn.Conv2d):
        # Kaiming (He) initialization for convolutional layers
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    
    elif isinstance(m, nn.LayerNorm):
        # Initialize LayerNorm weights and biases to ones and zeros, respectively
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    
    elif isinstance(m, nn.Embedding):
        # Uniform initialization for embedding layers
        nn.init.uniform_(m.weight, -0.1, 0.1)


if __name__ == "__main__":
    train_loader = create_train_loader(
        train_dir, 
        batch_size=batch_size, 
        val_split=0.2, 
    )
    
    val_loader = create_val_loader(
        train_dir, 
        batch_size=batch_size, 
        val_split=0.2, 
    )

    device = torch.device('cuda')
    cuda_available = torch.cuda.is_available()
    print(f"Is CUDA available? {cuda_available}")
    model = AlzheimerModel(in_channels, patch_size, embed_size, img_size, num_layers, num_heads, d_mlp, dropout_rate)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Scaler for mixed precision
    scaler = GradScaler('cuda')

    if START_EPOCH > 0:
        checkpoint_path = f'output/param/checkpoint{START_EPOCH}.pth'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f'Loaded model from {checkpoint_path}')
    else:
        # Apply the weight initialization for first epoch
        print("Initialising weights for the first time")
        model.apply(initialize_weights)

    


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    for epoch in range(START_EPOCH, NUM_EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)  # Calculate validation loss

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

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
        #torch.save(model.state_dict(), f'output/param/alzheimer_vit_epoch_{epoch+1}.pth')
        if epoch < 10 or (epoch+1) % 20 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                #'optimizer_state_dict': optimizer.state_dict(),
            }, f'output/param/checkpoint{epoch+1}.pth')
            print(f'Model saved: checkpoint{epoch+1}.pth')

    