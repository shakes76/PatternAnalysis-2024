from modules import AlzheimerModel
from dataset import create_data_loader, IMAGE_DIM, NUM_PATCHES, D_MODEL
import torch
import torch.optim as optim
import torch.nn as nn


# Parameters
EPOCHS = 10
train_dir = 'dataset/AD_NC/train'
test_dir = 'dataset/AD_NC/test'
batch_size = 32
learning_rate = 0.0001
num_epochs = 10

# Model configuration
num_layers = 6 
num_heads = 8
d_mlp = 512
head_layers = 256
dropout_rate = 0.1

if __name__ == "__main__":

    train_loader = create_data_loader(train_dir, batch_size=batch_size, train=True)
    test_loader = create_data_loader(test_dir, batch_size=batch_size, train=False)

    model = AlzheimerModel(
        num_patches=NUM_PATCHES,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=D_MODEL,
        d_mlp=d_mlp,
        head_layers=head_layers,
        dropout_rate=dropout_rate,
        num_classes=2
    )
    device = torch.device('cuda')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy after epoch {epoch+1}: {accuracy:.2f}%')

        # Save the model checkpoint after each epoch
        torch.save(model.state_dict(), f'/output/paramalzheimer_vit_epoch_{epoch+1}.pth')
        print(f'Model saved: alzheimer_vit_epoch_{epoch+1}.pth')