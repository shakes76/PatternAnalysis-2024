# train.py
import torch
import torch.nn as nn
from modules import UNet3D
class Trainer:
    def __init__(self, train_loader):
        # Initialize the 3D U-Net model
        self.model = UNet3D(in_channels=4, out_channels=6, init_features=32)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # Define optimizer and loss function
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        # Set the training data loader
        self.train_loader = train_loader

    def train(self, n_epochs=1):
        for epoch in range(n_epochs):
            print(f"Epoch {epoch + 1}")
            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:  # Unpack the tuple
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                # Compute the loss (using labels as targets instead of images)
                labels_class_indices = torch.argmax(labels, dim=1)
                loss = self.criterion(outputs, labels_class_indices)  
                # Zero the gradients
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            print(running_loss/len(self.train_loader))
        
        return self.model
