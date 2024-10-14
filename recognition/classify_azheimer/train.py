import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset import get_dataloaders
from tqdm import tqdm
from modules import GFNet

class AlzheimerClassifier:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    def train(self, train_loader, epochs=10):
        self._bar = tqdm(range(epochs*len(train_loader)))
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                self._bar.update(1)
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Accuracy: {accuracy}")
        return accuracy


def main():
    data_dir = "recognition/classify_azheimer/AD_NC"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GFNet(num_classes=2)  # Assuming GFNet is designed for binary classification
    classifier = AlzheimerClassifier(model, device)
    train_loader, test_loader = get_dataloaders(data_dir)
    classifier.train(train_loader, epochs=10)
    accuracy = classifier.evaluate(test_loader)
    if accuracy >= 0.8:
        print("Model meets the accuracy requirement.")
    else:
        print("Model does not meet the accuracy requirement.")

if __name__ == "__main__":
    main()