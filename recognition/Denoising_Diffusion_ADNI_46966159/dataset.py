import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# specify directory of data
# dataroot = "/home/groups/comp3710/ADNI"
dataroot = "/home/lgmoak/Nextcloud/University/Courses/COMP3710/Assessment/PatternAnalysis-2024/recognition/Denoising_Diffusion_ADNI_46966159/ADNI"

batch_size = 28
image_size = 64

dataset = torchvision.datasets.ImageFolder(root=dataroot,
                                           transform=transforms.Compose([
                                               transforms.Resize((image_size, image_size)),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

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

def show_first_batch(loader):
    """
    Show training images from first batch
    """
    for batch in loader:
        show_images(batch[0], "Images in the first batch")
        break


if __name__ == '__main__':
    show_first_batch(dataloader)
