import torch
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from modules import GNNModel
from dataset import load_data

def train():
    data, classes = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = GNNModel(in_channels=data.num_features, hidden_channels=64, out_channels=len(classes)).to(device)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    losses = []
    train_accuracies = []
    val_accuracies = []
    epochs = 200

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Evaluate training set
        model.eval()
        _, pred = out.max(dim=1)
        train_correct = pred[data.train_mask] == data.y[data.train_mask]
        train_acc = int(train_correct.sum()) / int(data.train_mask.sum())
        train_accuracies.append(train_acc)

        # Evaluate validation set
        val_correct = pred[data.val_mask] == data.y[data.val_mask]
        val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
        val_accuracies.append(val_acc)

        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Training Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}')

    # Plot loss and accuracy curves
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('loss.png')

    plt.figure()
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig('accuracy.png')

    # Save model
    torch.save(model.state_dict(), 'model.pth')

    # Evaluate test set
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    print(f'Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    train()
