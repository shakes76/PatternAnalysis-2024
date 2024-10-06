from dataset import upload_dataset
from modules import GCNModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

device = torch.cpu
# Check if the CUDA && MPS for our laptop is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Cuda is available.")
else:
    print("CPU usage.")

# Upload our dataset
tensor_edges, tensor_targets, tensor_features = upload_dataset()

num_nodes = tensor_targets.shape[0]
node_indices = torch.arange(num_nodes)
scaler = StandardScaler()
tensor_features_cpu = tensor_features.cpu().numpy()
normalized_features = scaler.fit_transform(tensor_features_cpu)
tensor_features = torch.tensor(normalized_features, dtype=torch.float32).to(device)

# Check the results and numbers of those tensors
print("Data of edges: \n", tensor_edges)
print("Data of targets: \n", tensor_targets)
print("Data of features: \n", tensor_features)

num_samples_edges = tensor_edges.shape[0]
num_samples_targets = tensor_targets.shape[0]
num_samples_features = tensor_features.shape[0]

print(f'Number of samples in tensor_edges: {num_samples_edges}')
print(f'Number of samples in tensor_targets: {num_samples_targets}')
print(f'Number of samples in tensor_features: {num_samples_features}')

# Define the assignment of training, testing and cv set
train_id, test_id = train_test_split(node_indices, test_size=0.8, random_state=42)

train_features = tensor_features[train_id]
train_targets = tensor_targets[train_id]

test_features = tensor_features[test_id]
test_targets = tensor_targets[test_id]

train_dataset = TensorDataset(train_features, train_targets)
test_dataset = TensorDataset(test_features, test_targets)

print(f'Train features shape: {train_features.shape}')
print(f'Train targets shape: {train_targets.shape}')

# Load our data
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

num_classes = len(torch.unique(tensor_targets))
num_features = tensor_features.size(1)

# Training GCN model, using Adam as optimizer
number = len(torch.unique(tensor_targets))
model = GCNModel(classes=number).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()
def evaluate_accuracy(model, test_set, edge_index, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient computation needed
        for features, targets in test_set:
            features, targets = features.to(device), targets.to(device)

            # Forward pass
            outputs = model(features, edge_index)

            # Get predicted classes
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)  # Number of targets
            correct += (predicted == targets).sum().item()  # Count correct predictions

    accuracy = correct / total  # Calculate accuracy
    return accuracy

def train_model(model, train_loader, edges, criterion, optimizer, device, num_epochs, test_loader):
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features, edges)  # Pass edge_index to the model

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        accuracy = evaluate_accuracy(model, test_loader, edges, device)
        # Print training loss and accuracy
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}'
              f', Test Accuracy: {accuracy:.4f}')


# Train the model
train_model(model, train_loader, tensor_edges, criterion, optimizer, device, num_epochs=80, test_loader=train_loader)
