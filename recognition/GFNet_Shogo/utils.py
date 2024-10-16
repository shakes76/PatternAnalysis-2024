'''
To define helper method

Created by: Shogo Terashima
'''
import numpy as np
import torch

class EarlyStopping:
    def __init__(self, path='checkpoint.pt', min_delta=0.0, patience=5):
        '''
        Initialize the early stopping class.
        Args:
            min_deleta: Minimum change considered as improvement
            patience (int): specify how many consecutive times the condition is met before stopping.
        '''
        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        self.path = path
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, validation_loss, model):
        '''
        Check the early stopping conditions based on train and validation loss.
        Stop if overfitting.
        Stop if losses are converged.
        Args:
            model
            validation_loss (float): Current validation loss.
        '''
        score = -validation_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            
        # Stop training if patience is exceeded
        if self.counter >= self.patience:
            print(f"Early stopping triggered after {self.counter} consecutive epochs.")
            self.stop_training = True

    def save_checkpoint(self, model):
        '''
        Save model
        '''
        torch.save(model.state_dict(), self.path)
        self.best_model_state = model.state_dict()