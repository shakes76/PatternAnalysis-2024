from torch.utils.data import DataLoader
from dataset import ADNIDataset
from modules import GFNet, GFNetPyramid
import torch
from functools import partial
import torch.nn as nn


def accuracy(model, dataloader):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    # # Load the model
    # model.load_state_dict(torch.load('model.pth', weights_only=True))
    
    # # Set the model to evaluation
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device).float()
            labels = labels.to(device).long()
            
            preds = model(images)
            
            
            
            _, predicted = torch.max(preds, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total

def get_accuracy_from_path(model_path):
    
    model = GFNetPyramid(
            img_size=256, num_classes=2, in_chans=1,
            patch_size=4, embed_dim=[96, 192, 384, 768], depth=[3, 3, 27, 3],
            mlp_ratio=[4, 4, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.4, init_values=1e-6
        )
    
    model.load_state_dict(torch.load(model_path))
    
    acc = accuracy(model)
    return acc

    
    

def main():
    model = GFNet(
            img_size=256,
            patch_size=16, embed_dim=384, depth=12, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=2, in_chans=1
        )
    
    acc = accuracy(model)
    print(f'Accuracy: {acc}')
    
if __name__ == '__main__':
    main()