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
    

class WeightedMSELoss(nn.Module):
    def __init__(self, background_weight=0.1, brain_weight=1.0, edge_weight=2.0, edge_threshold=0.1):
        super().__init__()
        self.background_weight = background_weight
        self.brain_weight = brain_weight
        self.edge_weight = edge_weight
        self.edge_threshold = edge_threshold

    def create_weight_map(self, target):
        # Create a basic brain mask
        brain_mask = (target > self.edge_threshold).float()

        # Create an edge map using a simple gradient method
        grad_x = torch.abs(F.conv2d(target, torch.tensor([[[[1, -1]]]], device=target.device, dtype=target.dtype)))
        grad_y = torch.abs(F.conv2d(target, torch.tensor([[[[1], [-1]]]], device=target.device, dtype=target.dtype)))
        edge_map = (grad_x + grad_y > self.edge_threshold).float()

        # Combine masks
        weight_map = torch.ones_like(target) * self.background_weight
        weight_map = torch.where(brain_mask == 1, self.brain_weight, weight_map)
        weight_map = torch.where(edge_map == 1, self.edge_weight, weight_map)

        return weight_map

    def forward(self, pred, target):
        weight_map = self.create_weight_map(target)
        return torch.mean(weight_map * (pred - target) ** 2)
