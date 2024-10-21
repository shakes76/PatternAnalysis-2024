"""
Util.py

Contains various Utility functions. Currently handles rotation of matricies.
"""
import torch
import numpy as np

def rotate(tensor:torch.Tensor, dim0:int=0, dim1:int=1) -> torch.Tensor:
    if dim0 == dim1:
        return tensor
    tensor = tensor.transpose(dim0, dim1)
    tensor = tensor.flip(dim1)
    return tensor

# List of all possible unique rotation combinations. Can be used to chose a random unique rotation
UNIQUE_ROTATION_COMBOS = [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 2, 0), (0, 2, 1), (2, 0, 0), (2, 0, 1), (0, 3, 0), (0, 3, 1), (0, 3, 2), (1, 3, 0), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3), (1, 2, 0), (1, 2, 1), (3, 0, 0), (3, 0, 1)]
