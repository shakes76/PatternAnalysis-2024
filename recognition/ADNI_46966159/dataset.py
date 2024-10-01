import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# specify directory of data
dataroot = "/home/lgmoak/Nextcloud/University/Courses/COMP3710/Assessment/PatternAnalysis-2024/recognition/ADNI_46966159/ADNI"
# dataroot = "/home/groups/comp3710/ADNI"

batch_size = 28
n_epochs = 200
epsilon = 0.001
image_size = 256

dataset = torchvision.datasets.ImageFolder(root=dataroot,
                                           transform=transforms.Compose([
                                               transforms.CenterCrop(image_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

# pick device
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

    # sub-plots
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

