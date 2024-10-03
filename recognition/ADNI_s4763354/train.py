import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from GFNet.gfnet import GFNet, GFNetPyramid
writer = SummaryWriter()

sys.path.append(os.path.abspath('/home/Student/s4763354/comp3710/GFNet'))

from modules import GFNetBinaryClassifier
from dataset import get_data_loaders

def train_and_evaluate(train_dir, test_dir, epochs=10, lr=1e-5, batch_size=64):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    #train_loader, test_loader = get_data_loaders(train_dir, test_dir, batch_size=batch_size)
    train_loader, val_loader, test_loader = get_data_loaders(train_dir, test_dir, batch_size=batch_size)

    # # Fetch a single batch from the training loader
    # data_iter = iter(train_loader)
    # images, labels = next(data_iter)
    
    # print(f'Image batch shape: {images.shape}')  # [batch_size, 3, 224, 224]
    # print(f'Label batch shape: {labels.shape}')  # [batch_size]
    # print(f'First label in the batch: {labels[0]}') 

    # Load the pretrained model
    model = GFNetPyramid(img_size=224, patch_size=4, num_classes=1000)  # Original number of classes
    
    # Load the state dict and extract the model weights
    state_dict = torch.load("gfnet-h-ti.pth")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    
    # Load the weights, ignoring mismatched keys
    model.load_state_dict(state_dict, strict=False)
    
    # Modify the final layer for binary classification
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, 2)  # Change to 2 classes
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Training loop
    global_step = 0 
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Log training loss for each batch
            writer.add_scalar("Loss/train", loss.item(), global_step)
            global_step += 1

        # Compute training accuracy and loss
        train_accuracy = 100 * correct / total
        train_losses.append(running_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_accuracies.append(val_accuracy)

        # Log training accuracy and validation accuracy for each epoch
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%')
    
    writer.flush()
    writer.close()

    # Testing loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    # Plot training loss and accuracy
    # plt.figure(figsize=(12, 4))
    
    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label='Training Loss')
    # plt.title('Loss over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig("train_loss.png") 
    # plt.close() 
    
    # plt.subplot(1, 2, 2)
    # plt.plot(train_accuracies, label='Training Accuracy')
    # plt.title('Accuracy over Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy (%)')
    # plt.legend()
    # plt.savefig("train_acc.png") 
    # plt.close() 
    
    # plt.show()
    
    # Save the trained model
    torch.save(model.state_dict(), 'gfnet_20.pth')

if __name__ == "__main__":
    train_and_evaluate(train_dir='/home/groups/comp3710/ADNI/AD_NC/train', 
                    test_dir='/home/groups/comp3710/ADNI/AD_NC/test', epochs=20)
