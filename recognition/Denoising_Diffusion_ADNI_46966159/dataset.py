import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# specify directory of data
# dataroot = "/home/groups/comp3710/ADNI"
dataroot = "/home/lgmoak/Nextcloud/University/Courses/COMP3710/Assessment/PatternAnalysis-2024/recognition/Denoising_Diffusion_ADNI_46966159/ADNI"

# chosen hyperparameters
batch_size = 128
image_size = 64

# normalise, resize images and randomly flip
dataset = torchvision.datasets.ImageFolder(root=dataroot,
                                           transform=transforms.Compose([
                                               transforms.Resize((image_size, image_size)),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.1798, 0.1798, 0.1798), (0.1964, 0.1964, 0.1964)),
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
    fig = plt.figure(figsize=(6, 5))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # sub-plots
    idx = 0
    for i in range(rows):
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):

                ax.imshow(images[idx][0], cmap="gray")
                ax.axis('off')
                idx += 1
    fig.suptitle(title, fontsize=30)

    plt.show()

def show_first_batch(loader):
    """
    Show training images from first batch
    """
    for batch in loader:
        show_images(batch[0], "Images in the first batch")
        break

def show_forward(ddpm, loader):
    """
    Visual forward process step
    :param ddpm: diffusion model
    :param loader: dataloader
    :return:
    """
    for batch in loader:
        imgs = batch[0]

        show_images(imgs, "Original images")

        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(
                ddpm(imgs.cpu(),
                     [int(percent * 1000) - 1 for _ in range(len(imgs))]),
                f"DDPM Noisy images {int(percent * 100)}%"
            )
        break

def calculate_mean_std():
    """
    Run once to calculate the mean and std of ADNI dataset
    :return: mean, std
    """
    dataset = torchvision.datasets.ImageFolder(root=dataroot,
                                               transform=transforms.Compose([
                                                   transforms.Resize((image_size, image_size)),
                                                   transforms.ToTensor(),
                                               ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

    # Calculate mean and std
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in tqdm(dataloader):
        batch_samples = images.size(0)  # batch size (number of images in batch)
        images = images.view(batch_samples, images.size(1), -1)  # Flatten the images
        mean += images.mean(2).sum(0)  # Sum over the batch
        std += images.std(2).sum(0)    # Sum over the batch
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean, std


if __name__ == '__main__':
    print(calculate_mean_std())
    show_first_batch(dataloader)