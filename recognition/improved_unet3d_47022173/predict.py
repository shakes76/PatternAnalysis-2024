import torch
from dataset import *
import torch
from torch.utils.data import DataLoader, random_split
import torchio as tio
from utils import *
from modules import *

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

if IS_RANGPUR:
    images_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs/"
    masks_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only/"
    epochs = 50
    batch_size = 4
else:
    images_path = "./data/semantic_MRs_anon/"
    masks_path = "./data/semantic_labels_anon/"
    epochs = 5
    batch_size = 2

# Model parameters
in_channels = 1 # greyscale
n_classes = 6 # 6 different values in mask
base_n_filter = 8

batch_size = batch_size
num_workers = 2

if __name__ == '__main__':
    transforms = tio.Compose([
        tio.Resize((128,128,128)),
    ])

    model = Modified3DUNet(in_channels, n_classes, base_n_filter)
    model.to(device)
    model.load_state_dict(torch.load('model.pth'))
    
    test_dataset = ProstateDataset3D(images_path, masks_path, transforms, "test")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    test_loss = 0.0
    dice_scores = [0] * n_classes 
    criterion = DiceLoss()

    with torch.no_grad():
        for i, data in enumerate(test_dataloader):  
            inputs, masks = data
            inputs, masks = inputs.to(device), masks.to(device)
            if masks.dtype != torch.long:
                masks = masks.long()

            # Forward pass
            softmax_logits, predictions, logits = model(inputs)

            masks = masks.view(-1)  # Flatten masks
            masks = F.one_hot(masks, num_classes=6) # [2097152, 6]

            # Group categories for masks and labels

            for i in range(n_classes):
                mask = masks[:, i]
                prediction = predictions[:, i]
                dice_scores[i] += criterion.dice_coefficient(prediction, mask)

    # Dice score
    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Average Dice Score: {list(map(lambda x: float(x / len(test_dataloader)), dice_scores))}")