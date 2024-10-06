
# Importing libraries
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# Dataset class
class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Args:
            root (str): The root directory of the dataset
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        """
        self.root = root 
        self.transform = transform

        self.files = sorted(os.listdir(root))

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.files[index]))

        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.files)



# Function to get the dataloader
def get_dataloader(root, image_size, batch_size): # Size of image (256,240)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  #Singel channel image
    ])

    dataset = ImageDataset(root, transform)

    return DataLoader(dataset, batch_size=batch_size)

# Function to get the image with transform

class ImageVisualize:
    def __init__(self, dataloader):
        """
        Initializes the ImageVisualizer with a given DataLoader.
        Args:
            dataloader (DataLoader): PyTorch DataLoader containing the images.
        """
        self.dataloader = dataloader

    def get_image(self):
        for _, img in enumerate(self.dataloader):
            return img

    def plot_image(self, img=None):
        to_pil = transforms.ToPILImage()
        
        if img is None:
            img = self.get_image()

        if len(img.size()) == 4:
            img = img[0]

        img = to_pil(img)
        img.show()

    def plot_image_with_matplotlib(self, img=None):
        """
        Plots the given image using matplotlib. If no image is provided, it retrieves one from the DataLoader.
        Args:
            img (Tensor, optional): A 3D or 4D image tensor to be plotted. If None, fetches from DataLoader.
        """
        # If no image is provided, retrieve one from the DataLoader
        if img is None:
            img = self.get_image()

        # If the image is 4D (batch, channels, height, width), use the first image in the batch
        if len(img.size()) == 4:
            img = img[0]  # Select the first image in the batch

        # Convert the 3D tensor to numpy array and rearrange dimensions (Channels, Height, Width) -> (Height, Width, Channels)
        img = img.permute(1, 2, 0).numpy()

        # If the image was normalized to [-1, 1], denormalize back to [0, 1]
        img = img * 0.5 + 0.5

        # Plot the image using matplotlib
        plt.imshow(img)
        plt.axis('off')
        plt.show()

