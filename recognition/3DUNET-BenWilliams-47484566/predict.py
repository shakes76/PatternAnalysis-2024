import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MRI3DDataset
from train import train_model
import modules


image_folder = 'semantic_MRs_anon'
label_folder = 'semantic_labels_anon'

# Load dataset and dataloader
dataset = MRI3DDataset(image_folder, label_folder, normImage=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize model, criterion, and optimizer
model = modules.UNet3D(in_channels=3, out_channels=3)
criterion = modules.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0]))
optimizer = optim.Adam(model.parameters(), lr=0.001)

dice_scores = train_model(model, dataloader, criterion, optimizer, num_epochs=25)

print(f"Final Dice scores after training: {dice_scores}")
