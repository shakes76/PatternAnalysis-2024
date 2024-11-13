# ====================================================
# File: train.py
# Description: Contains code for training the model, including configuration of hyperparameters,
#              model saving, and evaluation metrics logging.
# Author: Hemil Shah
# Date Created: 14-11-2024
# Version: 1.0
# License: MIT License
# ====================================================

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

def train(model, graph_data, train_mask, epochs=160):
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out, _ = model(graph_data)
        loss = criterion(out[train_mask], graph_data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
