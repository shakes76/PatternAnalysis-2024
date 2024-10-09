"""
containing the source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training
"""

from dataset import train_set
import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

for inputs, masks in train_set:
    # Display the original image
    plt.imshow(inputs[ :, :, 10], cmap='gray')
    plt.axis('off')  # Hide axes
    plt.show()

    plt.imshow(masks[ :, :, 10, 2], cmap='jet')
    plt.axis('off')  # Hide axes
    plt.show()

# print("Validation Set: ", validation_set)
