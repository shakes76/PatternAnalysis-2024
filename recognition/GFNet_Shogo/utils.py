'''
To define helper method

Created by: Shogo Terashima
'''
import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
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
        if self.best_loss is None:
            self.best_loss = validation_loss
            self.save_checkpoint(model)
        elif validation_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                self.early_stop = True
        else:
            self.best_loss = validation_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
        self.best_model_state = model.state_dict()
