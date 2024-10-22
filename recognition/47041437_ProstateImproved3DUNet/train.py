import os
import torch
import torch.nn as nn

################### Uncomment to test UNet3D
#from module_unet3D import UNet3D

from module_improvedunet3D import UNet3D
from dataset import augmentation, load_data_3D


class dice_loss(nn.Module):
    def __init__(self, smooth=0.1):
        super(dice_loss, self).__init__()
        self.smooth = smooth
   
    
    def label_loss(self, pedictions, targets, smooth=0.1):
        intersection = (pedictions * targets).sum()  
        total = pedictions.sum() + targets.sum()                         
        dice_coeff = (2.0 * intersection + smooth) / (total + smooth)  
        return dice_coeff
    
    # Calculate DSC for each channel, add them up and get the mean
    def forward(self, pedictions, targets, smooth=0.1):    
        
        # Predictions and targets for each label
        prediction_0 = (pedictions.argmax(1) == 0) 
        target_0 = (targets == 0) 

        pediction_1 = (pedictions.argmax(1) == 1) 
        target_1 = (targets == 1) 

        pediction_2 = (pedictions.argmax(1) == 2) 
        target_2 = (targets == 2) 

        pediction_3 = (pedictions.argmax(1) == 3) 
        target_3 = (targets == 3) 

        pediction_4 = (pedictions.argmax(1) == 4) 
        target_4 = (targets == 4) 

        pediction_5 = (pedictions.argmax(1) == 5) 
        target_5 = (targets == 5) 

        # Calculates DSC for each label
        label_0 = self.label_loss(prediction_0, target_0)
        label_1 = self.label_loss(pediction_1, target_1)
        label_2 = self.label_loss(pediction_2, target_2)
        label_3 = self.label_loss(pediction_3, target_3)
        label_4 = self.label_loss(pediction_4, target_4)
        label_5 = self.label_loss(pediction_5, target_5)
        
        # Total DSC averaged over all labels
        dice = (label_0 + label_1 + label_2 + label_3 + label_4 + label_5) / 6.0    
        
        return 1 - dice, {
        'Label 0': label_0,
        'Label 1': label_1,
        'Label 2': label_2,
        'Label 3': label_3,
        'Label 4': label_4,
        'Label 5': label_5,
    }

