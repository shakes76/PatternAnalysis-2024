from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Load image as RGB
        
        if self.transform:
            image = self.transform(image)

        return image, 0  # Return image and dummy label (0)

def data_set_creator(image_size, batch_size):
    augmentation_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  
        transforms.Resize((image_size, image_size)),  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # Update the directory to point to your dataset (folder containing images)
    data_dir = 'recognition/Style GAN - 47219647/AD_NC/'  # Ensure this path points to your folder of images
    
    # Use the custom dataset class to load all images from the folder
    dataset = CustomImageDataset(image_dir=data_dir, transform=augmentation_transforms)

    # Create a data loader with the given batch size
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    return data_loader, dataset
