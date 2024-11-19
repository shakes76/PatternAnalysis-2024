'''
Defines a helper class for early stopping during model training.
Class list: EarlyStopping

Created by: Shogo Terashima
Created by:     Shogo Terashima
ID:             S47779628
Last update:    25/10/2024
'''
import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    '''
    Implements early stopping to halt training when validation loss stops improving.
    '''
    def __init__(self, path='checkpoint.pt', min_delta=0.0, patience=5):
        '''
        Initialize the early stopping class.

        Args:
            min_delta: Minimum change considered as improvement
            patience (int): specify how many consecutive times the condition is met before stopping.
        '''
        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        self.path = path
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, validation_loss, model):
        '''
        Checks if the validation loss has improved and updates stopping criteria.

        Args:
            validation_loss (float): Current validation loss.
            model (torch.nn.Module): Model to save if validation loss improves.
        '''
        if self.best_loss is None:
            self.best_loss = validation_loss
            self.save_checkpoint(model)
        elif validation_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                self.early_stop = True
        else:
            # Reset counter if validation loss improves
            self.best_loss = validation_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''
        Saves the model's current state as a checkpoint.

        Args:
            model (torch.nn.Module): Model to save.
        '''
        torch.save(model.state_dict(), self.path)
        self.best_model_state = model.state_dict()
