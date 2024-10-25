import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from modules import GCN
from dataset import load_data

def train(model, data, optimizer, class_weights):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()
    return loss.item()

def accuracy(model, data, mask):
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        correct = pred[mask].eq(data.y[mask]).sum().item()
        acc = correct / int(mask.sum())
    return acc

def plot_loss(loss_values):
    plt.figure()
    plt.plot(loss_values, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

def main():
    data, class_weights, num_classes = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features=128, hidden_dim=256, num_classes=num_classes).to(device)
    data = data.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)

    best_acc = 0
    patience = 20
    patience_counter = 0
    loss_values = []

    for epoch in range(200):
        loss = train(model, data, optimizer, class_weights)
        loss_values.append(loss)

        train_acc = accuracy(model, data, data.train_mask)
        test_acc = accuracy(model, data, data.test_mask)

        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save best model
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f'Epoch: {epoch} Loss: {loss:.4f} Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')

    plot_loss(loss_values)
    print(f"Best Test Accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()
