import torch.optim as optim

def get_hyperparameters():
    """
    Return a dictionary containing all the hyperparameters for the model.
    """
    return {
        'learning_rate': 1e-5,          
        'batch_size': 32,                
        'num_epochs': 30,              
        'weight_decay': 1e-4,          
        'dropout_rate': 0.1,            
        'lr_scheduler_step_size': 7,   
        'lr_scheduler_gamma': 0.1       
    }

def get_optimizer(model, params):
    """
    Return the optimizer (AdamW) with weight decay.
    """
    return optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

def get_scheduler(optimizer, params):
    """
    Return a learning rate scheduler for the optimizer.
    """
    return optim.lr_scheduler.StepLR(optimizer, step_size=params['lr_scheduler_step_size'], gamma=params['lr_scheduler_gamma'])