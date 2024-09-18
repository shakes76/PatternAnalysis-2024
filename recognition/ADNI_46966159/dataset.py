import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# specify directory of data
dataroot = "/home/lgmoak/Nextcloud/University/Courses/COMP3710/Assessment/PatternAnalysis-2024/recognition/ADNI_46966159/ADNI"
# dataroot = "/home/groups/comp3710/ADNI"

batch_size = 128
n_epochs = 20
epsilon = 0.001
image_size = 256

dataset = torchvision.datasets.ImageFolder(root=dataroot,
                                           transform=transforms.Compose([
                                               transforms.Resize(image_size),
                                               transforms.CenterCrop(image_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

# Decide which device we want to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_images(images, title=""):
    """Plot input images"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for i in range(rows):
        for j in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()

# Plot some training images
real_batch = next(iter(dataloader))
show_images(real_batch[0], "Training")
