import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from dataset import get_dataloaders
import matplotlib.pyplot as plt

# Got inspiration from engine.py file of the following github repo:
# https://github.com/shakes76/GFNet
# And my train/evaluate code from the brain GAN code.

# def train_model(model, train_loader, val_loader, num_epochs=25):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    
#     train_loss, val_loss, train_acc, val_acc = [], [], [], []

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         epoch_loss = running_loss / len(train_loader)
#         epoch_acc = correct / total
        
#         train_loss.append(epoch_loss)
#         train_acc.append(epoch_acc)
        
#         # Validation step (similar to training loop but without optimizer step)
#         val_epoch_loss, val_epoch_acc = validate_model(model, val_loader, criterion)
#         val_loss.append(val_epoch_loss)
#         val_acc.append(val_epoch_acc)
        
#         print(f"Epoch: [{epoch+1}/{num_epochs}], Loss: {epoch_loss}, Accuracy: {epoch_acc}")

#     plot_metrics(train_loss, val_loss, train_acc, val_acc)
#     torch.save(model.state_dict(), "gfnet_model.pth")

# def validate_model(model, val_loader, criterion):
#     model.eval()
#     val_loss = 0.0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
    
#     avg_loss = val_loss / len(val_loader)
#     accuracy = correct / total
#     return avg_loss, accuracy

# def plot_metrics(train_loss, val_loss, train_acc, val_acc):
#     plt.figure(figsize=(10, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(train_loss, label='Train Loss')
#     plt.plot(val_loss, label='Validation Loss')
#     plt.title('Loss Over Time')
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     plt.plot(train_acc, label='Train Accuracy')
#     plt.plot(val_acc, label='Validation Accuracy')
#     plt.title('Accuracy Over Time')
#     plt.legend()
    
#     plt.show()
#     plt.savefig("/home/Student/s4743000/COMP3710/PatternAnalysis-2024/recognition/ADNI-GFNet-47430004/test/train/test_train", bbox_inches='tight', pad_inches=0)
#     plt.close()
