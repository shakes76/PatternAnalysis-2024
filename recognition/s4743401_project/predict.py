import torch
import torch.nn as nn
from modules import VisionTransformer
from dataset import dataloader
from train import train_model
import torch.optim as optim
device = torch.device("mps")
data_dir = '/Users/gghollyd/comp3710/report/AD_NC/'
model_path = '/Users/gghollyd/comp3710/report/module_weights.pth'

if __name__ == '__main__':
    # Set up model
    device = 'mps' 
    train_loader, test_loader, class_names = dataloader(data_dir)
    model = VisionTransformer(
                                num_layers=8,
                                img_size=(240, 256),  # Non-square image, height=240, width=256
                                emb_size=768,         # Embedding dimension
                                patch_size=16,        # Patch size (16x16 patches)
                                num_head=6,           # Number of attention heads
                                num_class=10          # Number of output classes
                            ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, optimizer, criterion, 1)
    #model.load_state_dict(torch.load(model_path, weights_only=True))
    #model.eval()  # Set the model to evaluation mode
    #dataloaders, class_names = dataloader(data_dir)