import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # List of all JPEG images in the directory
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith((".jpeg", ".jpg"))]

        # Print for debugging purposes
        print(f"Found {len(self.image_paths)} JPEG images in {self.image_dir}.")
        if len(self.image_paths) == 0:
            print(f"Warning: No JPEG files found in {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Greyscale
        if self.transform:
            image = self.transform(image)
        return image

# Define transformations for greyscale images
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to desired size
    transforms.ToTensor(),  # Convert to tensor (1 channel)
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

# Specify the correct directories
train_dir = './AD/train'
test_dir = './AD/test'

# Create datasets for training and testing
train_dataset = ImageDataset(image_dir=train_dir, transform=image_transforms)
test_dataset = ImageDataset(image_dir=test_dir, transform=image_transforms)

# Create DataLoader for both datasets
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Check if data loads correctly
for images in train_loader:
    print(f'Train Batch size: {images.shape}')  # Should be [batch_size, 1, height, width]
    break

for images in test_loader:
    print(f'Test Batch size: {images.shape}')  # Should be [batch_size, 1, height, width]
    break
