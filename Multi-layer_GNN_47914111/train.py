import torch
import numpy as np
import matplotlib.pyplot as plt
from modules import GCN  
import dataset 

# Call the preprocessing function and assign returned data
adjacency_matrix, train_mask, validation_mask, test_mask, feature_matrix, labels = dataset.preprocess_adjacency_matrix()

# Initialize the data dictionary with the returned values
data = {
    'features': feature_matrix,
    'adjacency_matrix': adjacency_matrix,
    'labels': labels,
    'train_mask': train_mask,
    'validation_mask': validation_mask,
    'test_mask': test_mask
}

# Initialize GCN model
model = GCN(feature_matrix.size(1), 16, len(torch.unique(labels)), 0.5)


def train_epoch(model, optimizer, criterion, data, clip=1.0):
    # 1. Set the model to training mode
    model.train()
    
    # 2. Clear the previous gradients
    optimizer.zero_grad()

    # 3. Forward pass, compute outputs
    outputs = model(data['features'], data['adjacency_matrix'])

    # 4. Compute the training loss
    train_loss = criterion(outputs[data['train_mask']], data['labels'][data['train_mask']])

    # 5. Backward pass and update model weights
    train_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    
    optimizer.step()

    # 6. Compute the validation loss
    validation_loss = evaluate_validation_loss(model, criterion, data)

    # 7. Return the training and validation losses
    return train_loss, validation_loss


def evaluate_validation_loss(model, criterion, data):
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():  # Disable gradient computation
        outputs = model(data['features'], data['adjacency_matrix'])
        validation_loss = criterion(outputs[data['validation_mask']], data['labels'][data['validation_mask']])

    return validation_loss


def test_accuracy(model, mask, data):
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        # Forward pass, compute outputs
        outputs = model(data['features'], data['adjacency_matrix'])

        # Get model predictions
        predictions = torch.argmax(outputs, dim=1)

        # Select correct predictions based on the mask
        correct_predictions = (predictions[mask] == data['labels'][mask])

        # Compute accuracy
        accuracy = correct_predictions.sum().item() / mask.sum().item()

    return accuracy


def train(model, data, epochs=100, patience=5, clip=1.0):
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = torch.nn.CrossEntropyLoss()

    train_acc_list, validation_acc_list = [], []
    train_loss_list = []  # Initialize training loss list
    validation_loss_list = []  # Initialize validation loss list

    best_validation_acc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        #Training
        train_loss, validation_loss = train_epoch(model, optimizer, criterion, data, clip=clip)

        #Validation
        validation_acc = test_accuracy(model, data['validation_mask'], data)
        train_acc = test_accuracy(model, data['train_mask'], data)

        # Save losses and accuracies
        train_acc_list.append(train_acc)
        validation_acc_list.append(validation_acc)
        train_loss_list.append(train_loss.item())
        validation_loss_list.append(validation_loss.item())

        # Get the current learning rate and print
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d}, LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, Validation Acc: {validation_acc:.4f}")

        # Learning rate scheduler: Adjust learning rate based on validation accuracy
        scheduler.step(validation_acc)

        # Early stopping mechanism
        if validation_acc > best_validation_acc:
            best_validation_acc = validation_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}, best validation accuracy: {best_validation_acc:.4f}")
            break

    # Evaluate the final test accuracy
    test_acc = test_accuracy(model, data['test_mask'], data)
    print(f"Test Accuracy: {test_acc:.4f}, Best Validation Accuracy: {best_validation_acc:.4f}")

    # Plot training and validation accuracy, loss
    plot_metrics(train_acc_list, validation_acc_list, train_loss_list, validation_loss_list)

    # Return training and validation accuracy, loss, and test accuracy
    return train_acc_list, validation_acc_list, train_loss_list, validation_loss_list, test_acc


def plot_metrics(train_acc_list, validation_acc_list, train_loss_list, validation_loss_list):
    # Accuracy plot
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, len(train_acc_list) + 1), train_acc_list, label="Train Accuracy")
    plt.plot(np.arange(1, len(validation_acc_list) + 1), validation_acc_list, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend(loc="lower right")
    plt.savefig("C:/Users/Wangyucheng/Desktop/comp3710a3/PatternAnalysis-2024/Multi-layer_GNN_47914111/plots/gcn_accuracy.png")
    plt.show()

    # Loss plot
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, len(train_loss_list) + 1), train_loss_list, label="Train Loss")
    plt.plot(np.arange(1, len(validation_loss_list) + 1), validation_loss_list, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend(loc="upper right")
    plt.savefig("C:/Users/Wangyucheng/Desktop/comp3710a3/PatternAnalysis-2024/Multi-layer_GNN_47914111/plots/gcn_loss.png")
    plt.show()


# Train the model and plot metrics
train(model, data, epochs=100, patience=10)

# Save the trained model
torch.save(model.state_dict(), "GCN_Model.pt")
