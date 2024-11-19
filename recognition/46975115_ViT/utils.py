import torch.optim as optim

def get_hyperparameters():
    """
    Returns a dictionary of hyperparameters used for training the model.

    Returns:
        dict: A dictionary containing:
            - 'batch_size' (int): Number of samples per batch.
            - 'num_epochs' (int): Number of epochs to train the model.
            - 'learning_rate' (float): Learning rate for the optimizer.
    """
        
    return {
        'batch_size': 64,
        'num_epochs': 50,
        'learning_rate': 1e-4,
    }

def get_optimizer(model, params):
    """
    Creates an Adam optimizer with the specified hyperparameters.

    Args:
        model (nn.Module): The neural network model to optimize.
        params (dict): Dictionary of hyperparameters, including the learning rate.

    Returns:
        optim.Adam: An Adam optimizer configured with the given learning rate and weight decay.
    """
    
    # Initialize the Adam optimizer with learning rate and weight decay
    return optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-4)
