
# Importing libraries
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import torch


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
    
    def get_raw_image(self, index):
        """
        Returns the original, untransformed image.
        
        Args:
            index (int): Index of the image in the dataset.

        Returns:
            PIL.Image: Raw image without any transformations.
        """
        img_path = os.path.join(self.root, self.files[index])
        raw_img = Image.open(img_ath).convert("L")  # Always return the raw image in RGB format
        return raw_img
    
    def compare(self, index):
        img = Image.open(os.path.join(self.root, self.files[index]))
        # img_path = os.path.join(self.root, self.files[index])
        # raw_img = Image.open(img_path).convert("L")  # Always return the raw image in RGB format
        
        if self.transform:
            trans_img = self.transform(img)
            to_pil = transforms.ToPILImage()
            trans_img = to_pil(trans_img)

        else:
            trans_img = None  # No transformation applied

        return img, trans_img,

def get_transform(image_size=(256, 240)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  #Singel channel image
    ])
    return transform

# Function to get the dataloader
def get_dataloader(root, image_size, batch_size, shuffle = False): # Size of image (256,240)
    transform = get_transform(image_size)

    dataset = ImageDataset(root, transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

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
    def denormalize(self, img):
        """
        Denormalizes the given image tensor from [-1, 1] to [0, 1].
        Args:
            img (Tensor): Image tensor to be denormalized.
        Returns:
            Tensor: Denormalized image tensor.
        """
        return img * 0.5 + 0.5
    
    def demour_image(self,img):
        """
        Denormalizes the given image tensor from [-1, 1] to [0, 255].
        Args:
            img (Tensor): Image tensor to be denormalized.
        Returns:
            Tensor: Denormalized image tensor in the range [0, 255].
        """
        return (img * 0.5 + 0.5) * 255
    
    def compare_img(self, img=None):

        if img is None:
            img = self.get_image()
        # # Convert tensor to PIL image if necessary
        # if isinstance(img, torch.Tensor):
        #     to_pil = ToPILImage()
        #     original_image = to_pil(img[0])  # Convert the first image in the batch to PIL
        # else:
        #     original_image = img
        
        to_pil = transforms.ToPILImage()
        og_img = self.denormalize(img)
        original_image = to_pil(og_img[0])
        


        transform = get_transform()
        transformed_image = transform(original_image)

        # Convert the transformed tensor back to a PIL image if needed for display
        transformed_image_pil = to_pil(transformed_image) if isinstance(transformed_image, torch.Tensor) else transformed_image

    
        # Display the original and transformed image
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Convert the transformed tensor back to a PIL image for visualization
        transformed_image_pil = to_pil(transformed_image)

        axes[1].imshow(transformed_image_pil)
        axes[1].set_title("Transformed Image")
        axes[1].axis('off')
        plt.show()

    def compare2_img(self, img=None):
        # Get an image tensor from the DataLoader or use the provided one
        if img is None:
            img = self.get_image()

        # Convert tensor to PIL image if necessary
        if isinstance(img, torch.Tensor):
            to_pil = ToPILImage()
            original_image = to_pil(img[0])  # Convert the first image in the batch to PIL
        else:
            original_image = img

        # Define a sample transformation pipeline (can be customized)
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor()  # Convert back to Tensor for visualization if needed
        ])
        
        # Apply transformations to the PIL image
        transformed_image = transform(original_image)

        # Convert the transformed tensor back to a PIL image if needed for display
        transformed_image_pil = to_pil(transformed_image) if isinstance(transformed_image, torch.Tensor) else transformed_image

        # Display the original and transformed images side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Show the original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Show the transformed image
        axes[1].imshow(transformed_image_pil)
        axes[1].set_title("Transformed Image")
        axes[1].axis('off')

        plt.show()

    def display_comparison(original, transformed):
        """
        Display the original and transformed images side-by-side.
        
        Args:
            original (PIL.Image): The original image.
            transformed (PIL.Image or None): The transformed image (or None if not available).
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Display the original image
        axes[0].imshow(original)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Display the transformed image if available
        if transformed:
            axes[1].imshow(transformed)
            axes[1].set_title("Transformed Image")
        else:
            axes[1].text(0.5, 0.5, "No Transformation Applied", fontsize=15, ha='center')
        axes[1].axis('off')

        plt.show()
