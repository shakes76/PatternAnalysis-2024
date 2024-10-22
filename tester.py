import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Function to convert labels to one-hot encoded channels
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    # Get unique values (assuming categorical data)
    channels = np.unique(arr)
    # Create a result array with channels as the first dimension
    res = np.zeros((len(channels),) + arr.shape, dtype=dtype)
    
    # Loop over each unique category (channel)
    for c in channels:
        c = int(c)
        # Set the corresponding channel to 1 where the value matches the category
        res[c] = (arr == c).astype(dtype)
    
    return res

# Function to load 3D data
def load_data_3D(image_names, norm_image=False, categorical=False, dtype=np.float32, early_stop=False): 
    num = len(image_names)
    first_case = nib.load(image_names[0]).get_fdata(caching='unchanged')
    
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # Remove extra dim
    
    if categorical:
        # Convert to categorical (one-hot) encoding with channels as the first dimension
        first_case = to_channels(first_case, dtype=dtype)
        channels, depth, height, width = first_case.shape
        images = np.zeros((num, channels, depth, height, width), dtype=dtype)
    else:
        depth, height, width = first_case.shape
        images = np.zeros((num, 4, depth, height, width), dtype=dtype)  # Assuming 4 channels

    for i, image_name in enumerate(tqdm(image_names)):
        try:
            nifti_image = nib.load(image_name)
            in_image = nifti_image.get_fdata(caching='unchanged')

            if len(in_image.shape) == 4:
                in_image = in_image[:, :, :, 0]  # Remove extra dim
            in_image = in_image.astype(dtype)

            if norm_image:
                in_image = (in_image - in_image.mean()) / in_image.std()  # Normalization
            
            if categorical:
                in_image = to_channels(in_image, dtype=dtype)
                images[i, :, :in_image.shape[1], :in_image.shape[2], :in_image.shape[3]] = in_image
            else:
                images[i, :, :in_image.shape[0], :in_image.shape[1], :in_image.shape[2]] = in_image

            if early_stop and i > 20:
                break
        except FileNotFoundError as e:
            print(f"Error loading image: {image_name}. {e}")

    return images
# Custom dataset class for PyTorch with transformations for 3D data
class CustomDataset(Dataset):
    def __init__(self, image_filenames, img_dir, labels_dir, transform=None):
        self.image_filenames = image_filenames
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        image = load_data_3D([img_path])[0]  # Assuming load_data_3D returns a 3D image (depth, height, width)
        image = torch.tensor(image, dtype=torch.float32)

        label_filename = self.image_filenames[idx].replace("LFOV", "SEMANTIC_LFOV")  # replacement
        label_path = os.path.join(self.labels_dir, label_filename)
        label = load_data_3D([label_path], categorical=True)[0]
        label = torch.tensor(label, dtype=torch.long)  # Use long for class indices
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        return image, label

# Define custom transformations for 3D data
class Compose3D:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

class Resize3D:
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def __call__(self, volume):
        return np.resize(volume, self.new_shape)

# UNet3D model definition
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet3D, self).__init__()

        features = init_features
        self.encoder1 = UNet3D._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = UNet3D._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = UNet3D._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder4 = UNet3D._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet3D._block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet3D._block(features * 16, features * 8)
        
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet3D._block(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet3D._block(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet3D._block(features * 2, features)

        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.softmax(self.conv(dec1), dim=1)

    @staticmethod
    def _block(in_channels, features):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

# Main training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_dir = os.path.join("Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-", "data", 
                        "HipMRI_study_complete_release_v1", "semantic_MRs_anon")
labels_dir = os.path.join("Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-", "data", 
                        "HipMRI_study_complete_release_v1", "semantic_labels_anon")

# Check if the directories exist
if not os.path.exists(img_dir):
    print(f"Image directory {img_dir} does not exist.")
if not os.path.exists(labels_dir):
    print(f"Labels directory {labels_dir} does not exist.")

# Proceed with data loading if paths are correct
image_filenames = [f for f in os.listdir(img_dir) if f.endswith('.nii.gz')]

dataset = CustomDataset(image_filenames, img_dir, labels_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = UNet3D(in_channels=4, out_channels=6)  # Change out_channels to the number of classes
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class segmentation
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(10):
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch + 1}/10], Loss: {loss.item():.4f}")
