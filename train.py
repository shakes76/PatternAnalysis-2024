import torch
from sklearn.metrics import accuracy_score
from dataset import load_facebook_data
from modules import GNNModel


data = load_facebook_data('facebook.npz')

# Split the data into training, validation, and test sets

# First 80% of nodes for training
train_mask = torch.randperm(data.num_nodes)[:int(0.8 * data.num_nodes)]
# Next 10% for validation
val_mask = torch.randperm(data.num_nodes)[int(0.8 * data.num_nodes):int(0.9 * data.num_nodes)]
# Last 10% for testing
test_mask = torch.randperm(data.num_nodes)[int(0.9 * data.num_nodes):]

# GNN model
model = GNNModel(input_dim=128, hidden_dim=64, output_dim=data.y.max().item() + 1)

# Define optimizer and loss function with L2 regularization
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# Cross-entropy loss for multi-class classification
loss_fn = torch.nn.CrossEntropyLoss()

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Early stopping criteria
best_val_loss = float('inf')
# Stop training if validation loss doesn't improve for 10 epochs
patience = 10
patience_counter = 0

# Training
def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients
    out = model(data)  # Forward pass through the model
    loss = loss_fn(out[train_mask], data.y[train_mask])  # Calculate loss using the training mask
    loss.backward()  # Backpropagate the gradients
    optimizer.step()  # Update the model weights


    #  loss value for monitoring
    return loss.item()

# Validation and testing function
def test():
    model.eval()
    with torch.no_grad():
        out = model(data)  # Forward pass through the model
        val_loss = loss_fn(out[val_mask], data.y[val_mask]).item()  #  validation loss

        # Get the predicted class for test set
        test_pred = out[test_mask].argmax(dim=1)

        #  test accuracy
        test_acc = accuracy_score(data.y[test_mask].cpu(), test_pred.cpu())
    return val_loss, test_acc  # Return the validation loss and test accuracy

# Train the model for up to 200 epochs
for epoch in range(200):
    loss = train()  # Run a single training step
    scheduler.step()  # Update learning rate

    val_loss, test_acc = test()  # Evaluate the model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Test Accuracy: {test_acc:.4f}')

# Save the final trained model
torch.save(model.state_dict(), 'gnn_model.pth')
