import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from modules import GNNModel , EnhancedGNN
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import load_facebook_data, split_data
import matplotlib.pyplot as plt

def train_model(data, model, epochs=400, learning_rate=0.001, weight_decay=1e-4):
    """
    This function trains the given model on the given data.
    The training loop is run for the specified number of epochs.
    The model is trained using the Adam optimiser with the given learning rate.
    The learning rate is reduced by a factor of 0.4 if the validation accuracy does not improve for 7 epochs.
    The training loss and validation accuracy are tracked for each epoch and plotted at the end.

    Parameters:
    -----------
    data : torch_geometric.data.Data
        The input data for the model
    model : torch.nn.Module
        The model to train
    epochs : int, optional (default=400)
        The number of epochs to train the model for
    learning_rate : float, optional (default=0.001)
        The learning rate for the Adam optimiser
    weight_decay : float, optional (default=1e-4)
        The weight decay for the Adam optimiser
    
    Returns:
    --------
    model : torch.nn.Module
        The trained model

    """
    # Prepare masks
    train_mask, val_mask, test_mask = split_data(data)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay= weight_decay)

    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # learning rate scheduler
    scheduler = ReduceLROnPlateau(optimiser, mode='max', factor=0.4, patience=7, verbose=True)

    # Track metrics
    train_losses = []
    val_accuracies = []

    # training loop
    for epoch in range(epochs):
        model.train()
        # forward pass
        optimiser.zero_grad()
        out = model(data.x, data.edge_index)

        # loss
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])

        # backward pass
        loss.backward()
        # optimiser step
        optimiser.step()

        # track losses
        train_losses.append(loss.item())

        # validation
        model.eval()

        with torch.no_grad():
            pred = out.argmax(dim=1)
            correct = pred[data.val_mask] == data.y[data.val_mask]
            val_acc = int(correct.sum()) / int(data.val_mask.sum())
            val_accuracies.append(val_acc)

        # early stopping
        if len(val_accuracies) > 5:
            if val_accuracies[-1] < val_accuracies[-2] < val_accuracies[-3] < val_accuracies[-4] < val_accuracies[-5]:
                break

        # print (each value upto 4 deicmal places)
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")

    # PLotting
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_accuracies, label='Val Acc')
    plt.legend()
    plt.show()

    return model