import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import CNN

class TrainCNN:
    def __init__(self, num_classes, input_width, input_height, learning_rate=0.001):
        self.model = CNN(num_classes, input_width, input_height)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self, train_loader: DataLoader, num_epochs: int):
        self.model.train() 
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
            avg_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    def evaluate(self, test_loader: DataLoader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))