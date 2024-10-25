# Reference Link:
# https://www.kaggle.com/code/mrmohammadi/2d-unet-pytorch


import numpy as np
import nibabel as nib
from tqdm import tqdm

#Remember to remove early_stop=True

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1

    return res


# load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage : bool (normalize the image 0.0-1.0)
    early_stop : Stop loading prematurely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
            # inImage = inImage / np.linalg.norm(inImage)
            # inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            #inImage = utils.to_channels(inImage, dtype=dtype)
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, orient=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

    normImage : bool (normalize the image 0.0-1.0)
    orient : Apply orientation and resample image? Good for images with large slice thickness or anisotropic resolution
    dtype : Type of the data. If dtype=np.uint8, it is assumed that the data is labels
    early_stop : Stop loading prematurely? Leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # interp = 'continuous'
    interp = 'linear'
    if dtype == np.uint8:  # assume labels
        interp = 'nearest'

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    if orient:
        niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
        # testResultName = "oriented.nii.gz"
        # niftiImage.to_filename(testResultName)
    first_case = niftiImage.get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if orient:
            niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth]  # clip slices
        inImage = inImage.astype(dtype)
        if normImage:
            # inImage = inImage / np.linalg.norm(inImage)
            # inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            #inImage = utils.to_channels(inImage, dtype=dtype)
            inImage = to_channels(inImage, dtype=dtype)
            # images[i, :, :, :, :] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2], :inImage.shape[3]] = inImage  # with pad
        else:
            # images[i, :, :, :] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2]] = inImage  # with pad

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

# Staff linked this page regarding resizing images
# https://stackoverflow.com/questions/64674612/how-to-resize-a-nifti-nii-gz-medical-image-file
import skimage.transform as skTrans

# Define the target shape
target_shape = (256, 128)

"""
# Function to resize images using Scikit-Image
def resize_images_skimage(images, target_shape):
    # Create an empty array to store the resized images
    #                         ((num images    ,      256       ,        128     )
    resized_images = np.zeros((images.shape[0], target_shape[0], target_shape[1]), dtype=images.dtype)
    
    # Loop through each image in the dataset
    for i, image in enumerate(images):
        # Resize the image to the target shape using Scikit-Image's resize function
        resized_image = skTrans.resize(image, target_shape, order=1, preserve_range=True, anti_aliasing=True)

        # Store the resized image in the resized_images array
        resized_images[i, :, :] = resized_image

    return resized_images
"""

# Function to resize a single image
def resize_image(image, target_shape):
    return skTrans.resize(image, target_shape, order=1, preserve_range=True, anti_aliasing=True)

def pad_channels(image, target_channels):
    """
    Pad the channels of an image to the target number of channels.
    """
    current_channels = image.shape[-1]
    
    # If the current number of channels is less than the target, pad with zeros
    if current_channels < target_channels:
        padding_shape = list(image.shape)
        padding_shape[-1] = target_channels - current_channels
        padding = np.zeros(padding_shape, dtype=image.dtype)
        image = np.concatenate((image, padding), axis=-1)

    return image

# Function to load and resize images one by one using load_data_2D
def load_and_resize_images(image_paths, target_shape, normImage=False, categorical=False, target_channels=6):
    resized_images = []  # To store resized images

    for image_path in image_paths:
        # Load image one at a time using load_data_2D
        image = load_data_2D([image_path], normImage=normImage, categorical=categorical, early_stop=True)  # Loading one image at a time
        resized_image = resize_image(image[0], target_shape)  # Resize the single image

        # If categorical, pad the channels to the target number of channels
        if categorical:
            resized_image = pad_channels(resized_image, target_channels)

        resized_images.append(resized_image)  # Append the resized image to the list

    # Stack all resized images into a NumPy array
    return np.stack(resized_images)

import os
# Define the root directory
dataroot = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/"


# Create paths for images and segmentation labels
train_image_dir = os.path.join(dataroot, "keras_slices_train")
train_label_dir = os.path.join(dataroot, "keras_slices_seg_train")

val_image_dir = os.path.join(dataroot, "keras_slices_validate")
val_label_dir = os.path.join(dataroot, "keras_slices_seg_validate")

test_image_dir = os.path.join(dataroot, "keras_slices_test")
test_label_dir = os.path.join(dataroot, "keras_slices_seg_test")

# Get all image and label file paths
train_image_paths = sorted([os.path.join(train_image_dir, f) for f in os.listdir(train_image_dir) if f.endswith('.nii.gz')])
train_label_paths = sorted([os.path.join(train_label_dir, f) for f in os.listdir(train_label_dir) if f.endswith('.nii.gz')])

val_image_paths = sorted([os.path.join(val_image_dir, f) for f in os.listdir(val_image_dir) if f.endswith('.nii.gz')])
val_label_paths = sorted([os.path.join(val_label_dir, f) for f in os.listdir(val_label_dir) if f.endswith('.nii.gz')])

test_image_paths = sorted([os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if f.endswith('.nii.gz')])
test_label_paths = sorted([os.path.join(test_label_dir, f) for f in os.listdir(test_label_dir) if f.endswith('.nii.gz')])

# Load data
train_images_resized = load_and_resize_images(train_image_paths, target_shape, normImage=True, categorical=False)
val_images_resized = load_and_resize_images(val_image_paths, target_shape, normImage=True, categorical=False)
test_images_resized = load_and_resize_images(test_image_paths, target_shape, normImage=True, categorical=False)

# Segmentation Masks
train_labels_resized = load_and_resize_images(train_label_paths, target_shape, normImage=False, categorical=True)
val_labels_resized = load_and_resize_images(val_label_paths, target_shape, normImage=False, categorical=True)
test_labels_resized = load_and_resize_images(test_label_paths, target_shape, normImage=False, categorical=True)

"""
#debug
print(f"Resized shape of train_images: {train_images_resized.shape}")
print(f"Resized shape of train_labels: {train_labels_resized.shape}")
print(f"Resized shape of val_images: {val_images_resized.shape}")
print(f"Resized shape of val_labels: {val_labels_resized.shape}")
print(f"Resized shape of test_images: {test_images_resized.shape}")
print(f"Resized shape of test_labels: {test_labels_resized.shape}")
"""

"""
def find_max_channels(image_paths):
    max_channels = 0
    for image_path in image_paths:
        # Load the image (segmentation mask)
        image = load_data_2D([image_path], normImage=False, categorical=False)  # Load without one-hot encoding
        unique_labels = np.unique(image[0])  # Find unique labels in the mask
        max_channels = max(max_channels, len(unique_labels))  # Track the maximum number of unique labels
    return max_channels

# Check maximum number of channels in the training labels
max_train_channels = find_max_channels(train_label_paths)
print(f"Maximum number of channels in training labels: {max_train_channels}")

# Check maximum number of channels in the validation labels
max_val_channels = find_max_channels(val_label_paths)
print(f"Maximum number of channels in validation labels: {max_val_channels}")

# Check maximum number of channels in the test labels
max_test_channels = find_max_channels(test_label_paths)
print(f"Maximum number of channels in test labels: {max_test_channels}")
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def up_block(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Encoder (Reduced depth)
        self.conv1 = conv_block(1, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)

        # Bottleneck (Remove)
        self.bottleneck = conv_block(256, 512)

        # Decoder (Reduced depth)
        self.upconv3 = up_block(512, 256)
        self.conv3_1 = conv_block(512, 256)

        self.upconv2 = up_block(256, 128)
        self.conv2_1 = conv_block(256, 128)

        self.upconv1 = up_block(128, 64)
        self.conv1_1 = conv_block(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, 6, kernel_size=1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(nn.MaxPool2d(2)(conv1))
        conv3 = self.conv3(nn.MaxPool2d(2)(conv2))
        conv4 = self.conv4(nn.MaxPool2d(2)(conv3))

        # Bottleneck
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(conv4))

        # Decoder
        upconv4 = self.upconv4(bottleneck)
        conv4_1 = self.conv4_1(torch.cat([upconv4, conv4], dim=1))

        upconv3 = self.upconv3(conv4_1)
        conv3_1 = self.conv3_1(torch.cat([upconv3, conv3], dim=1))

        upconv2 = self.upconv2(conv3_1)
        conv2_1 = self.conv2_1(torch.cat([upconv2, conv2], dim=1))

        upconv1 = self.upconv1(conv2_1)
        conv1_1 = self.conv1_1(torch.cat([upconv1, conv1], dim=1))

        output = self.out_conv(conv1_1)
        return output


# Convert ===Resized=== NumPy arrays to PyTorch tensors
train_images_tensor = torch.Tensor(train_images_resized).unsqueeze(1)  # Add channel dimension
train_labels_tensor = torch.Tensor(train_labels_resized).unsqueeze(1)  # Add channel dimension

val_images_tensor = torch.Tensor(val_images_resized).unsqueeze(1)
val_labels_tensor = torch.Tensor(val_labels_resized).unsqueeze(1)

test_images_tensor = torch.Tensor(test_images_resized).unsqueeze(1)  # Add channel dimension for test images
test_labels_tensor = torch.Tensor(test_labels_resized).unsqueeze(1)  # Add channel dimension for test labels


# Create DataLoaders for batching
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Initialize the UNet model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Previous assignment used
# device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
model = UNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        """
         # Print original labels shape
        print(f"Original labels shape: {labels.shape}")

        # Step 1: Remove the extra dimension with squeeze (if present)
        labels_squeezed = torch.squeeze(labels, dim=1)  # If shape is [8, 1, 256, 128], it becomes [8, 256, 128, 6]
        print(f"After squeeze, labels shape: {labels_squeezed.shape}")

        # Step 2: Convert one-hot encoded labels to class indices
        labels_argmax = torch.argmax(labels_squeezed, dim=2)  # Now, it should collapse the last dimension to [8, 256, 128]
        print(f"After argmax, labels shape: {labels_argmax.shape}")

        # You can also check if the operation correctly modified the tensor by verifying some of the values
        print(f"Sample labels (argmax): {labels_argmax[0, :, :]}")  # Print the class indices for the first image
        """

        # Remove extra dimension
        labels = torch.squeeze(labels, dim=1)

        # Convert one-hot encoded labels to class indices (Required by Cross Entropy Loss)
        labels = torch.argmax(labels, dim=3)
        
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Debugging: print output and label shapes
        #print(f"Outputs shape: {outputs.shape}")
        #print(f"Labels shape: {labels.shape}")

        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    # Validation step (optional but recommended)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)

            val_labels = torch.squeeze(val_labels, dim=1)  # Remove extra dimension, shape becomes [8, 256, 128, 6]
            val_labels = torch.argmax(val_labels, dim=3)   # Convert one-hot encoding to class indices, shape becomes [8, 256, 128]

            outputs = model(val_images)
            loss = criterion(outputs, val_labels)
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader)}")


def dice_coefficient(pred, target, smooth=1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()

    if pred_flat.sum() == 0 and target_flat.sum() == 0:
        return 1.0  # If both are empty, Dice score is 1
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

# Evaluate on the test set
model.eval()
dice_scores = []

with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

        outputs = model(test_images)  # Add batch dimension here
        outputs = (outputs > 0.5).float()  # Threshold to binary mask

        dice = dice_coefficient(outputs, test_labels.unsqueeze(0))  # Add batch dimension here
        dice_scores.append(dice.item())

# Calculate average Dice coefficient
average_dice = np.mean(dice_scores)
print(f"Average Dice Coefficient: {average_dice}")