# train.py

import torch
from torch.optim import Adam
from torch_geometric.data import Data
from modules import GNN
from dataset import load_data, edges_path, features_path, labels_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
features, edge_index, labels, page_type_mapping = load_data(edges_path, features_path, labels_path)
data = Data(x=features, edge_index=edge_index, y=labels).to(device)


# Define the GNN model with updated architecture
input_dim = features.shape[1]
hidden_dim = 1024
output_dim = len(page_type_mapping)
model = GNN(input_dim, hidden_dim, output_dim).to(device)

# Split nodes into training and validation sets
from sklearn.model_selection import train_test_split
train_idx, val_idx = train_test_split(range(data.num_nodes), test_size=0.2, random_state=42)
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[train_idx] = True
data.val_mask[val_idx] = True

# Define optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Validation function
def validate():
    model.eval()
    with torch.no_grad():
        out = model(data)
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    return val_loss.item()

# Train the model
num_epochs = 2000
losses = []
val_losses = []

for epoch in range(num_epochs):
    loss = train()
    val_loss = validate()
    losses.append(loss)
    val_losses.append(val_loss)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}')

# Save the model after training
model_path = "/content/drive/My Drive/COMP3710/Project/gnn_model_weights.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Plotting the training and validation loss
import matplotlib.pyplot as plt

plt.plot(range(len(losses)), losses, label='Training Loss')
plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()