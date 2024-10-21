"""
Code for training, validating, tesing and saving the model.
@author Anthony Ngo
@date 21/10/2024
"""
import torch
from dataset import GNNDataLoader
from modules import GCNModel
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# hyperparams
lr = 0.01 # default
decay = 0.01 # default
epochs = 300

# loading data
data = GNNDataLoader(filepath='facebook.npz')

# data splits
# precaclulate permutation for consistent splits and no data leakages
perm = torch.randperm(data.num_nodes)
# we do a 80/10/10 split between training, validation and testing sets
train_label = perm[:int(0.8*data.num_nodes)] # 0 -> 80
valid_label = perm[int(0.8 * data.num_nodes):int(0.9 * data.num_nodes)] # 80 -> 90
test_label = perm(data.num_nodes)[int(0.9 * data.num_nodes):] # 90 -> 100

# now we can define the model
# note here that the output dimensions is defined to be 4 classes:
    # politicians, governmental organizations, television shows and companies
model = GCNModel(input_dim=128, hidden_dim=64, output_dim=data.y.max().item() + 1) # +1 as labels start from 0

optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay) # using Adam optimiser
criterion = torch.nn.CrossEntropyLoss() 

# Potential Improvements
# LR scheduler

# implementing early stopping to reduce change of overfitting model
best_val_loss = float('inf')
patience_count = 0
patience_lim = 10 #stop training if loss doesn't improve for 10 epochs


# To plot
losses = []
valid_losses = []
accuracies = []

def _train_model():
    """
    The function for training the model using the pre-defined parameters and loss functions
    """
    model.train() # set model to training mode
    optimiser.zero_grad() # clear gradients from last backprop

    out = model(data) # forward pass
    loss = criterion(out[train_label], data.y[train_label])

    loss.backward() # backprop the gradients
    optimiser.step() # update params

    return loss.item() # return loss to monitor

def _test_model():
    """
    Function for testing the model accuracy against the validation and test data set
    """
    model.eval() # set model to eval mode

    with torch.no_grad():
        out = model(data)
        valid_loss = criterion(out[valid_label], data.y[valid_label]).item() # we calculate validation loss

        # get highest probability predictions for testing data
        predictions = out[test_label].argmax(dim=1)

        # calc accuracy by comparing predicted and true labels
        accuracy = accuracy_score(data.y[test_label].cpu(), predictions.cpu())

    return valid_loss, accuracy

def training_loop():
    for epoch in range(epochs):
        loss = _train_model()

        valid_loss, test_accuracy = _test_model()

        losses.append(loss)
        valid_losses.append(valid_loss)
        accuracies.append(test_accuracy)
        
        if valid_loss < best_val_loss:
            # early stoppage
            best_val_loss = valid_loss
            patience_count = 0
            torch.save(model.state_dict(), 'early_stop_model.pth')
        else:
            patience_count += 1

        if patience_count >= patience_lim:
            print(f'Stopping early at epoch {epoch}, patience reached')
            break
        # printing loss information each epoch
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss:.4f}, Val Loss: {valid_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # After training loop, generate loss plot.
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Num Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.show()

    torch.save(model.state_dict(), 'GCN_model.pth')