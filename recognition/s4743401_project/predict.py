import torch
from modules import VisionTransformer
from dataset import dataloader
device = torch.device("mps")
data_dir = '/Users/gghollyd/comp3710/report/AD_NC/'
# Save model after training
model_path = '/Users/gghollyd/comp3710/report/module_weights.pth'

if __name__ == '__main__':
    # Set up model
    model = VisionTransformer(img_size=224, patch_size=16, in_channels=1, num_classes=2)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set the model to evaluation mode
    dataloaders, class_names = dataloader(data_dir)