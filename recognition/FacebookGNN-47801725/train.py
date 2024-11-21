import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from modules import GCN 
from data import load_data, split_data  

def train(model, data, optimizer, criterion):
    """
    Train the GCN model for one epoch.

    Args:
        model (torch.nn.Module): The GCN model.
        data (torch_geometric.data.Data): The graph data object.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (torch.nn.Module): Loss function.

    Returns:
        float: Training loss.
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, mask, criterion):
    """
    Evaluate the GCN model on a given dataset split.

    Args:
        model (torch.nn.Module): The GCN model.
        data (torch_geometric.data.Data): The graph data object.
        mask (torch.Tensor): The mask indicating the data split (train, val, or test).
        criterion (torch.nn.Module): Loss function.

    Returns:
        tuple: Loss and accuracy.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = criterion(out[mask], data.y[mask])
        pred = out.argmax(dim=1)
        accuracy = accuracy_score(data.y[mask].cpu(), pred[mask].cpu())
    return loss.item(), accuracy

if __name__ == '__main__':
    # Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and split the dataset
    file_path = '/Users/eaglelin/Downloads/facebook.npz'  # Update with your dataset path
    data = load_data(file_path)
    data = split_data(data)
    data = data.to(device)

    # Define model parameters
    input_dim = data.num_features
    hidden_dims = [100, 64, 32]
    output_dim = len(torch.unique(data.y))

    # Initialize the model, optimizer, and loss function
    model = GCN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, dropout_rate=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
   
    # Lists to store loss and accuracy values
    train_losses = []
    val_losses = []
    val_accuracies = []
    test_accuracies = []
   
    # Training loop
    num_epochs = 200
    best_val_acc = 0
    for epoch in range(1, num_epochs + 1):
        # Train the model for one epoch
        train_loss = train(model, data, optimizer, criterion)
        # Evaluate the model on the validation set
        val_loss, val_acc = evaluate(model, data, data.val_mask, criterion)
        # Evaluate the model on the test set
        test_loss, test_acc = evaluate(model, data, data.test_mask, criterion)

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_gcn_model.pth')
       
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)

        # Print training, validation, and test statistics
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

    # Load the best model for final evaluation
    model.load_state_dict(torch.load('best_gcn_model.pth'))
    test_loss, test_acc = evaluate(model, data, data.test_mask, criterion)
    print(f'Best Test Accuracy: {test_acc:.4f}')

    epochs = range(1, num_epochs + 1)
    
   
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    
    plt.figure()
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation and Test Accuracy')
    plt.legend()
    plt.show()