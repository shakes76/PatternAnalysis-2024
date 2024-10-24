import torch

def train_model(model, dataloader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = correct / total * 100
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return model

