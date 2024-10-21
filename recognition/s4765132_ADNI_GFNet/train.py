import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from modules import GFNet 
from dataset import train_loader, test_loader  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = GFNet(num_classes=2).to(device)  
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_and_validate(model, train_loader, test_loader, num_epochs=10):
    train_loss_history = []
    train_acc_history = []
    test_acc_history = []

    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)
        
        test_acc = correct_test / total_test
        test_acc_history.append(test_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')

    
    plt.figure()
    plt.plot(train_loss_history, label='Train Loss')
    plt.title('Train Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(test_acc_history, label='Test Accuracy')
    plt.title('Train and Test Accuracy')
    plt.legend()
    plt.show()
    
    torch.save(model.state_dict(), 'alzheimer_gfnet.pth')


train_and_validate(model, train_loader, test_loader, num_epochs=10)
