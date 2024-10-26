import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from module import GCN
from dataset import load_data


# Accuracy function to calculate accuracy based on any mask (train or test)
def accuracy(model, data, mask):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        pred = model(data).argmax(dim=1)  # Predict class labels
        correct = pred[mask].eq(data.y[mask]).sum().item()  # Count correct predictions
        return correct / int(mask.sum())  # Return accuracy


# Training function
def train(model, data, optimizer, class_weights):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear previous gradients
    out = model(data)  # Forward pass
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights)  # Compute loss
    loss.backward()  # Backpropagate
    optimizer.step()  # Update weights
    return loss.item(), out  # Return the loss and output


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Load data
    edges_path = r"C:\Users\wuzhe\Desktop\musae_facebook_edges.csv"
    labels_path = r"C:\Users\wuzhe\Desktop\musae_facebook_target.csv"
    features_path = r"C:\Users\wuzhe\Desktop\musae_facebook_features.json"
    data, num_classes = load_data(edges_path, labels_path, features_path)

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(data.y.cpu()), y=data.y.cpu())
    class_weights = torch.tensor(class_weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Create masks for splitting data into 80% training, 10% validation, 10% test sets
    num_nodes = data.x.shape[0]
    train_mask = torch.rand(num_nodes) < 0.8  # 80% for training
    val_test_split = torch.rand(num_nodes)[~train_mask] < 0.5  # Split the remaining into validation and test
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[~train_mask] = val_test_split
    test_mask[~train_mask] = ~val_test_split

    data.train_mask = train_mask
    data.test_mask = test_mask

    # Set up model, device, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features=128, hidden_dim=256, num_classes=num_classes).to(device)
    data = data.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)

    train_loss_values = []
    test_loss_values = []
    train_acc_values = []
    test_acc_values = []
    best_acc = 0
    best_train_acc = 0

    for epoch in range(400):
        train_loss, out = train(model, data, optimizer, class_weights)  # Train the model
        train_loss_values.append(train_loss)

        # Calculate training accuracy
        train_acc = accuracy(model, data, data.train_mask)
        train_acc_values.append(train_acc)

        # Calculate test loss and accuracy
        model.eval()
        test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask], weight=class_weights).item()
        test_loss_values.append(test_loss)

        test_acc = accuracy(model, data, data.test_mask)
        test_acc_values.append(test_acc)

        # Track the best accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            best_train_acc = train_acc

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:3d}  Train Accuracy: {train_acc:.4f}  Train Loss: {train_loss:.4f}  Test Accuracy: {test_acc:.4f}  Test Loss: {test_loss:.4f}')
