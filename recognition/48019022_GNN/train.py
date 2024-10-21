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

"""
Potential Improvements:
    
"""

def _train_model(model, data, train):
    """
    The function for training the model using the pre-defined parameters and loss functions
    Although we pass the entire dataset through the forward pass, only the training indexes are updating the parameters
    """
    model.train() # set model to training mode
    optimiser.zero_grad() # clear gradients from last backprop

    out = model(data.to(device)) # forward pass
    loss = criterion(out[train].to(device), data.y[train].to(device))

    loss.backward() # backprop the gradients
    optimiser.step() # update params

    return loss.item() # return loss to monitor

def _evaluate_model(model, data, valid, test):
    """
    Function for testing the model accuracy against the validation and test data set
    """
    model.eval() # set model to eval mode

    with torch.no_grad():
        out = model(data)
        valid_loss = criterion(out[valid], data.y[valid]).item() # we calculate validation loss

        # get highest probability predictions for testing data
        predictions = out[test].argmax(dim=1)

        # calc accuracy by comparing predicted and true labels
        accuracy = accuracy_score(data.y[test].cpu(), predictions.cpu())

    return valid_loss, accuracy

def training_loop(num_epochs, model, data, train, valid, test):
    """
    Code for the main training loop.
    Data is passed through the network to train, and loss plots will be generated
    """
    # implementing early stopping to reduce change of overfitting model
    best_val_loss = float('inf')
    patience_count = 0
    patience_lim = 10 #stop training if loss doesn't improve for 10 epochs

    for epoch in range(num_epochs):
        loss = _train_model(model=model, data=data, train=train)

        # Validation and testing step
        valid_loss, test_accuracy = _evaluate_model(model=model, data=data, valid=valid, test=test)

        # Update stats
        losses.append(loss)
        valid_losses.append(valid_loss)
        accuracies.append(test_accuracy)
        
        # Step the scheduler
        scheduler.step()

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

if __name__ == '__main__':
    # Setting up CUDA
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    # Setting up environment:
    # initialise hyperparams
    lr = 0.01 # default
    decay = 0.01 # default
    epochs = 300 # default, perhaps try 200,400,500 etc

    # loading data
    data, train_idx, valid_idx, test_idx = GNNDataLoader(filepath='facebook.npz')

    data = data.to(device)
    # now we can define the model
    # note here that the output dimensions is defined to be 4 classes:
    # politicians, governmental organizations, television shows and companies
    model = GCNModel(input_dim=128, hidden_dim=64, output_dim=data.y.max().item() + 1) # +1 as labels start from 0
    model = model.to(device)
    model.train()

    # Setting up Adam Optimiser and Cross Entropy Loss function
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay) # using Adam optimiser
    criterion = torch.nn.CrossEntropyLoss()

    # Apply learning rate scheduler for better minima accuracy
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimiser, step_size=50, gamma=0.5)

    # Plotting Lists
    losses = []
    valid_losses = []
    accuracies = []

    training_loop(epochs, model=model, data=data, train=train_idx, valid=valid_idx, test=test_idx)