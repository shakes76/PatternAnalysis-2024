import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from modules import GCN, accuracy
from dataset import prepare_data, compute_class_weights, create_data


# Training function
def train(model, data, optimizer, class_weights):
    model.train()
    optimizer.zero_grad()  # Zero the gradients
    out = model(data)  # Forward pass
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update parameters
    return loss.item(), out


# Training loop function
def train_model(model, data, optimizer, class_weights, num_epochs=200, patience=20):
    # Store values for loss and accuracy
    train_loss_values = []
    test_loss_values = []
    train_acc_values = []
    test_acc_values = []
    best_acc = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        # Perform a training step
        train_loss, out = train(model, data, optimizer, class_weights)
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

        # Early stopping logic based on test accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at Epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:4d}  Train Accuracy: {train_acc:.4f}  Train Loss: {train_loss:.4f}  '
                  f'Test Accuracy: {test_acc:.4f}  Test Loss: {test_loss:.4f}')

    return train_loss_values, test_loss_values, train_acc_values, test_acc_values, best_acc


# Function to plot loss and accuracy curves
def plot_loss_accuracy(train_loss_values, test_loss_values, train_acc_values, test_acc_values):
    plt.figure(figsize=(12, 6))

    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_values, label='Train Loss', color='red')
    plt.plot(test_loss_values, label='Test Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_values, label='Train Accuracy', color='red')
    plt.plot(test_acc_values, label='Test Accuracy', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(42)

    # File paths for loading data
    edge_path = r"C:\Users\wuzhe\Desktop\musae_facebook_edges.csv"
    target_path = r"C:\Users\wuzhe\Desktop\musae_facebook_target.csv"
    features_path = r"C:\Users\wuzhe\Desktop\musae_facebook_features.json"

    # Load and prepare data
    node_features, edge_index, y, labels = prepare_data(edge_path, target_path, features_path)
    data = create_data(node_features, edge_index, y)

    # Set up device (GPU or CPU) and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_features=128, hidden_dim=256, num_classes=len(labels['page_type'].unique())).to(device)
    data = data.to(device)

    # Set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)

    # Compute class weights for handling class imbalance
    class_weights = compute_class_weights(labels)

    # Train the model
    train_loss_values, test_loss_values, train_acc_values, test_acc_values, best_acc = train_model(
        model, data, optimizer, class_weights)

    # Plot loss and accuracy curves
    plot_loss_accuracy(train_loss_values, test_loss_values, train_acc_values, test_acc_values)

    # Save the model
    torch.save(model.state_dict(), 'gcn_model.pth')
    print(f"Model saved. Best Test Accuracy: {best_acc:.4f}")
