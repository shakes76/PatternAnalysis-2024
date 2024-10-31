import torch
import torch.optim as optim
import torch.nn as nn
from modules import create_model
from dataset import get_dataloaders

def train_model(data_dir='/home/groups/comp3710/ADNI/AD_NC', num_epochs=20, batch_size=128, learning_rate=1e-4, num_classes=2, model_path='best_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders, dataset_sizes = get_dataloaders(data_dir, batch_size)
    model = create_model(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / dataset_sizes['train']
        train_acc = (running_corrects.double() / dataset_sizes['train']) * 100
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        test_loss = running_loss / dataset_sizes['test']
        test_acc = (running_corrects.double() / dataset_sizes['test']) * 100
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_path)

        print(f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}%")