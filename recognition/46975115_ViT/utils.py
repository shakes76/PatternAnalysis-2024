import torch.optim as optim

def get_hyperparameters():
    return {
        'batch_size': 64,
        'num_epochs': 25,
        'learning_rate': 1e-4,
    }

def get_optimizer(model, params):
    return optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=1e-4)
