# -*- coding: utf-8 -*-
"""
Dice similarity coefficient

Adapted from Keras code from [1] into PyTorch.

References:
    [1] Todayisagreatday, “Dice coefficent not increasing for U-net image segmentation,” Stack Overflow. 
    Accessed: Oct. 18, 2024. [Online]. Available: https://stackoverflow.com/q/67018431

@author: al
"""

import torch

def dice_coeff(pred, true, dev):
    # flatten pred from (6, x, y) to (1, x, y)
    predTmp = torch.permute(pred, (1, 2, 0))
    predTmp = torch.argmax(predTmp, dim = -1)
    predTmp = predTmp[None, :, :].to(dev)
    
    intersect = torch.sum(torch.sum(torch.abs(torch.mul(true, predTmp)), dim = 0))
    union = torch.sum(torch.sum(torch.add(torch.abs(true), torch.abs(predTmp)), dim = 0))
    return 2 * intersect / union