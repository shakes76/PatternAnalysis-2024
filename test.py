import torch
from sklearn.metrics import accuracy_score
from dataset import load_facebook_data
from modules import GNNModel


data = load_facebook_data('facebook.npz')

# Define the test mask (using the last 10% of nodes as the test set)
test_mask = torch.randperm(data.num_nodes)[int(0.9 * data.num_nodes):]

#  GNN model
model = GNNModel(input_dim=128, hidden_dim=64, output_dim=data.y.max().item() + 1)

# Load the previously saved model weights
model.load_state_dict(torch.load('gnn_model.pth'))

# Function to evaluate the model on the test set
def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        # Predict output for all nodes
        out = model(data)




        test_pred = out[mask].argmax(dim=1)  # Only get predictions for test set nodes
        test_acc = accuracy_score(data.y[mask].cpu(), test_pred.cpu())  # Calculate accuracy on test set
    return test_acc

# Evaluate the model on the test set
test_accuracy = evaluate(model, data, test_mask)
print(f'Test Accuracy: {test_accuracy:.4f}')
