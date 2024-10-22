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

def dice_coeff(pred, true, dev, lbls = 1):
    # flatten pred from (6, x, y) to (1, x, y)
    #predTmp = torch.permute(pred, (1, 2, 0))
    #predTmp = torch.argmax(predTmp, dim = -1)
    #predTmp = pred[None, :, :].to(dev)
    
    #testOnes = torch.full_like(pred, 6)
    #testOnes2 = torch.full_like(true, 6)
    #pred = testOnes
    #true = testOnes2
    
    intersect = torch.sum(torch.sum(torch.abs(true * pred), dim = 1))
    print("intersect = ", intersect.cpu().item())
    union = torch.sum(torch.sum(torch.add(torch.abs(true), torch.abs(pred)), dim = 1))
    print("union = ", union.cpu().item())
    return (2 * intersect / union) / lbls
