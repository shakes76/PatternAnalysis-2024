import torch
import numpy as np
import random
import argparse
from modules import UNet3D
from dataset import MRIDataset_pelvis
from torch.utils.data import DataLoader
import torch.nn as nn

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Load the model
model = UNet3D(in_channel=1, out_channel=6).cuda()
model.load_state_dict(torch.load(r'epoch_19_lossdice1.pth'))
model.eval()

# Define the test dataloader
test_dataset = MRIDataset_pelvis(mode='test', dataset_path=r'C:\Users\111\Desktop\3710\新建文件夹\数据集\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1')
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# Define weighted Dice loss function
class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=None, smooth=1):
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten the input and target tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = inputs.sum() + targets.sum()

        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (total + self.smooth)

        if self.weights is not None:
            return (1 - dice) * self.weights
        return 1 - dice

valid_loss = []
for idx, (data_x, data_y) in enumerate(test_dataloader):
    data_x = data_x.to(torch.float32).cuda()
    data_y = data_y.to(torch.float32).cuda().squeeze()

    # Get model outputs
    outputs = model(data_x)

    # Get the predicted class with the maximum value
    outputs_class = torch.argmax(outputs, dim=1).squeeze()

    # Calculate the intersection with the ground truth
    intersection = torch.sum(outputs_class == data_y)
    assert outputs_class.size() == data_y.size()

    # Calculate the Dice coefficient
    dice_coeff = intersection.item() / outputs_class.numel()
    print('Dice Coefficient:', dice_coeff)
    valid_loss.append(dice_coeff)

# Print the average Dice coefficient for the test set
average_loss = np.average(valid_loss)
print('Average Dice Coefficient:', average_loss)
