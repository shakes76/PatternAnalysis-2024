import torch
from torch import nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    def __init__(self, vae, threshold=0.5, weight=0.1):
        super().__init__()
        self.vae = vae
        self.vae.eval()
        self.threshold = threshold
        self.weight = weight
        for param in self.vae.parameters():
            param.requires_grad = False 
        
    def create_brain_mask(self, image):
        # Create a simple mask based on pixel intensity
        return (image > self.threshold).float()
    
    @torch.no_grad()
    def forward(self, input, target):
    
        if torch.isnan(input).any() or torch.isnan(target).any():
            print("NaN detected in input or target")
            return torch.tensor(0.0, requires_grad=True)
        
        input_mask = self.create_brain_mask(input)
        target_mask = self.create_brain_mask(target)
        
        input_encoding = self.vae.encode(input_mask)
        target_encoding = self.vae.encode(target_mask)
        
        if isinstance(input_encoding, tuple):
            input_encoding = input_encoding[0]
        if isinstance(target_encoding, tuple):
            target_encoding = target_encoding[0]

        return F.mse_loss(input_encoding, target_encoding)
    
