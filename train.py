import torch
from torch_geometric.utils import train_test_split_edges
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
from dataset import load_facebook_data
from modules import GNNModel

# Load data
data = load_facebook_data('facebook.npz')

# Split into training, validation, and test sets
train_mask = torch.randperm(data.num_nodes)[:int(0.8 * data.num_nodes)]
val_mask = torch.randperm(data.num_nodes)[int(0.8 * data.num_nodes):int(0.9 * data.num_nodes)]
test_mask = torch.randperm(data.num_nodes)[int(0.9 * data.num_nodes):]

# Initialize model
model = GNNModel(input_dim=128, hidden_dim=64, output_dim=data.y.max().item() + 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        out = model(data)
        val_loss = loss_fn(out[val_mask], data.y[val_mask]).item()
        test_pred = out[test_mask].argmax(dim=1)
        test_acc = accuracy_score(data.y[test_mask].cpu(), test_pred.cpu())
    return val_loss, test_acc

for epoch in range(200):  # Train for 200 epochs
    loss = train()
    if epoch % 10 == 0:
        val_loss, test_acc = test()
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'gnn_model.pth')

