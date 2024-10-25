import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from modules import GCN
from dataset import load_data

data, class_weights = load_data()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_features=128, hidden_dim=256, num_classes=len(torch.unique(data.y))).to(device)
data = data.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)

train_losses = []
test_accuracies = []
best_acc = 0

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

def accuracy(mask):
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        correct = pred[mask].eq(data.y[mask]).sum().item()
        acc = correct / int(mask.sum())
    return acc

for epoch in range(200):
    train()
    train_acc = accuracy(data.train_mask)
    test_acc = accuracy(data.test_mask)

    test_accuracies.append(test_acc)

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')

    if epoch % 10 == 0:
        print(f'Epoch: {epoch} Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')

plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.title('Training Loss and Test Accuracy over Epochs')
plt.legend()
plt.show()

model.load_state_dict(torch.load('best_model.pth'))
plot_umap(data, model)
print(f"Best Test Accuracy: {best_acc:.4f}")

