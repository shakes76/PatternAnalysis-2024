"""
Prepares all datasets specified into a format understandable by YOLOv11.
Converts masks into bounding boxes and downscales images for faster training and inference.

Usage example: python dataset.py -d test | to install all datasets except the test dataset

@author Ewan Trafford
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F  # Import functional transforms
import cv2
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from skimage import color
from skimage import img_as_float
import argparse
import yaml
import utils


class ISICDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        """
        Args:
            img_dir (str): Directory containing the images.
            mask_dir (str): Directory containing the corresponding masks.
            transform (callable, optional): Optional transform to be applied on both images and masks.
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_filenames = os.listdir(img_dir)  # List of image filenames
        
        # Ensure filenames are sorted to align images with masks
        self.img_filenames.sort()

    def __len__(self):
        """
        Returns the size of the specified dataset
        """
        return len(self.img_filenames)

    def __getitem__(self, idx):
        """
        Loads and returns both the image and mask
        """
        
        img_path = os.path.join(self.img_dir, self.img_filenames[idx])

        # Get the corresponding mask filename
        img_name = os.path.splitext(self.img_filenames[idx])[0]  # Extract the base filename (without extension)
        mask_filename = f"{img_name}_segmentation.png"  # Assuming the mask is in PNG format
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # Check if the mask file exists
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for image {self.img_filenames[idx]} at {mask_path}")

        # Load the image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Load mask as grayscale

        # Apply transformations (to both image and mask)
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask  # Return both the image and the mask

    def get_image_path(self, idx):
        """Retrieve the path of the image for the given index."""
        img_path = os.path.join(self.img_dir, self.img_filenames[idx])
        return img_path

    def get_mask_path(self, idx):
        """Retrieve the path of the mask for the given index."""
        img_name = os.path.splitext(self.img_filenames[idx])[0]  # Base filename
        mask_filename = f"{img_name}_segmentation.png"  # Assuming mask format
        mask_path = os.path.join(self.mask_dir, mask_filename)
        return mask_path


class SegmentationTransform:
    """
    Specifies transforms to be made to each dataset
    """
    def __init__(self, resize=(640, 640), normalize=True):
        self.resize = resize
        self.normalize = normalize

    def __call__(self, image, mask):
        # Resize both image and mask
        image = F.resize(image, self.resize)
        mask = F.resize(mask, self.resize, interpolation=Image.NEAREST)  # Use NEAREST for masks to preserve labels

        # Convert to tensor
        image = F.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()  # Convert mask to tensor with integer labels

        # Normalize image (do not normalize the mask)
        if self.normalize:
            image = F.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        return image, mask

def get_bounding_box(mask):
    """
    Get the bounding box of the non-zero regions in the mask.

    Args:
        mask (PIL Image): The mask image (should be in grayscale).

    Returns:
        tuple: (x_min, y_min, x_max, y_max) of the bounding box.
    """
    mask_np = np.array(mask)  # Convert mask to numpy array
    # Find the coordinates of non-zero pixels
    non_zero_pixels = np.argwhere(mask_np > 0)

    if non_zero_pixels.size == 0:
        return None  # No lesions found

    # Get the bounding box coordinates
    y_min, x_min = non_zero_pixels.min(axis=0)
    y_max, x_max = non_zero_pixels.max(axis=0)

    return (x_min, y_min, x_max, y_max)

def draw_bounding_box(image, bbox, color='green', thickness=3, buffer=8):
    """Draw a bounding box on the image with a buffer."""
    
    draw = ImageDraw.Draw(image)
    x_min, y_min, x_max, y_max = bbox

    # Apply buffer to the bounding box coordinates
    x_min = max(0, x_min - buffer)
    y_min = max(0, y_min - buffer)
    x_max = min(image.width, x_max + buffer)
    y_max = min(image.height, y_max + buffer)

    draw.rectangle([x_min, y_min, x_max, y_max], outline='green', width=thickness)


def overlay_mask_on_image(image, mask, alpha=0.45):
    """
    Overlay a translucent blue mask on an image, making the black areas of the mask transparent.

    Args:
        image (PIL Image): The original image to overlay on.
        mask (PIL Image): The mask image (should be in grayscale).
        alpha (float): Transparency level of the mask (0.0 to 1.0).
    """
    # Convert mask to a binary mask (0 for background, 255 for object)
    mask = mask.convert("L")  # Convert to grayscale
    mask_np = np.array(mask)  # Convert to numpy array

    # Create a color version of the mask (blue) with transparency
    colored_mask = np.zeros((*mask_np.shape, 4), dtype=np.uint8)  # RGBA
    colored_mask[..., :3] = [0, 255, 0]  # Set to green color
    # Set alpha channel: fully opaque for white areas, transparent for black
    colored_mask[..., 3] = (mask_np > 0) * int(255 * alpha)  # Set alpha based on mask

    # Convert colored mask back to PIL Image
    colored_mask_image = Image.fromarray(colored_mask, mode='RGBA')

    # Overlay the mask on the original image
    # Ensure the original image is also in RGBA format
    img_rgba = image.convert("RGBA")
    return Image.alpha_composite(img_rgba, colored_mask_image)

def resize_image(image):
    """
    Resizes a passed in image relative to its dimensions.

    Args:
        image (PIL Image): The original image to be resized.
    """
    if image.width > 4000 or image.height > 4000:   
        new_size = (image.width // 7, image.height // 7)
    elif image.width > 2000 or image.height > 2000:   
        new_size = (image.width // 4, image.height // 4)
    elif image.width > 1000 or image.height > 1000:
        new_size = (image.width // 2, image.height // 2)
    else:
        new_size = (image.width, image.height)

    # Resize the image
    img_resized = image.resize(new_size, Image.LANCZOS)

    return img_resized


def prepare_train():       
    """
    Prepares the training dataset in its respective directory
    """

    for i in range(0, train_dataset.__len__()):

        # Load a sample from the dataset
        sample_idx = i  # Change this to load different samples (up to 2593)
        image, mask = train_dataset[sample_idx]

        # Example usage
        image_path = train_dataset.get_image_path(sample_idx)
        mask_path = train_dataset.get_mask_path(sample_idx) 
        save_path = "Data/Training/images/image"+str(sample_idx)+".png"    
        label_save_path = "Data/Training/labels/image"+str(sample_idx)+".txt"

        # Open images
        image = Image.open(image_path)
        image = resize_image(image)
        mask = Image.open(mask_path)
        mask = resize_image(mask)

        bbox = get_bounding_box(mask)

        image.save(save_path)

        with open(label_save_path, 'w') as f:
            # Class ID is 0 for all images
            class_id = 0
            # Normalize the bounding box coordinates
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                width = image.width
                height = image.height
                
                # Convert to normalized coordinates
                centre_x = (x_min + x_max) / 2 / width
                centre_y = (y_min + y_max) / 2 / height
                norm_width = (x_max - x_min) / width
                norm_height = (y_max - y_min) / height
                    
                # Write the label to the file
                f.write(f"{class_id} {centre_x} {centre_y} {norm_width} {norm_height}\n")

        if (i % 100) == 0:
            print(str(i) + " images processed of " + str(train_dataset.__len__()))

    print ("Training dataset prepared")
    return

def prepare_val():
    """
    Prepares the validation dataset in its respective directory
    """

    print("Preparing validation data")

    for i in range(0, val_dataset.__len__()):

        # Load a sample from the dataset
        sample_idx = i 
        image, mask = val_dataset[sample_idx]

        # Example usage
        image_path = val_dataset.get_image_path(sample_idx)
        mask_path = val_dataset.get_mask_path(sample_idx) 
        save_path = "Data/Validation/images/image"+str(sample_idx)+".png"    
        label_save_path = "Data/Validation/labels/image"+str(sample_idx)+".txt"

        # Open images
        image = Image.open(image_path)
        image = resize_image(image)
        mask = Image.open(mask_path)
        mask = resize_image(mask)

        bbox = get_bounding_box(mask)

        image.save(save_path)

        with open(label_save_path, 'w') as f:
                # Class ID is 0 for all images
                class_id = 0
                # Normalize the bounding box coordinates
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    width = image.width
                    height = image.height
                    
                    # Convert to normalized coordinates
                    centre_x = (x_min + x_max) / 2 / width
                    centre_y = (y_min + y_max) / 2 / height
                    norm_width = (x_max - x_min) / width
                    norm_height = (y_max - y_min) / height
                    
                    # Write the label to the file
                    f.write(f"{class_id} {centre_x} {centre_y} {norm_width} {norm_height}\n")

        if (i % 100) == 0:
            print(str(i) + " images processed of " + str(val_dataset.__len__()))

    print ("Validation dataset prepared")

def prepare_test():
    """
    Prepares the testing dataset in its respective directory
    """

    print("Preparing testing data")

    for i in range(0, test_dataset.__len__()):

        # Load a sample from the dataset
        sample_idx = i
        image, mask = test_dataset[sample_idx]

        # Example usage
        image_path = test_dataset.get_image_path(sample_idx)
        mask_path = test_dataset.get_mask_path(sample_idx) 
        save_path = "Data/Testing/images/image"+str(sample_idx)+".png"    
        label_save_path = "Data/Testing/labels/image"+str(sample_idx)+".txt"

        # Open images
        image = Image.open(image_path)
        image = resize_image(image)
        mask = Image.open(mask_path)
        mask = resize_image(mask)

        bbox = get_bounding_box(mask)

        image.save(save_path)

        with open(label_save_path, 'w') as f:
                # Class ID is 0 for all images
                class_id = 0
                # Normalize the bounding box coordinates
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    width = image.width
                    height = image.height
                    
                    # Convert to normalized coordinates
                    centre_x = (x_min + x_max) / 2 / width
                    centre_y = (y_min + y_max) / 2 / height
                    norm_width = (x_max - x_min) / width
                    norm_height = (y_max - y_min) / height
                    
                    # Write the label to the file
                    f.write(f"{class_id} {centre_x} {centre_y} {norm_width} {norm_height}\n")

        if (i % 100) == 0:
            print(str(i) + " images processed of " + str(test_dataset.__len__()))

    print ("Testing dataset prepared")


def install_dataset(datasets):
    """
    Calculates what datasets to install based on passed in command line arguments.
    """
    if 'train' in datasets:
        prepare_train()
    if 'val' in datasets:
        prepare_val()
    if 'test' in datasets:
        prepare_test()
    print("All data processing finished")

def create_folder(name, path):
    """
    Creates an empty folder at the specified path with the given name.

    Args:
        name (string): The name of the new folder.
        path (string): Relative path to new folder.
    """

    print("Preparing " + name + " folder...")
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Folder "{name}" created.')
        if name in ('Training', 'Testing', 'Validation'):
            os.makedirs(path + "/images")
            os.makedirs(path + "/labels")
    else:
        print(f'Folder "{name}" already exists.')


def prepare_directories():
    """
    Creates all required folders and directories. Should only be called once.
    """

    create_folder('Data', 'Data')
    create_folder('Training', 'Data/Training')
    create_folder('Validation', 'Data/Validation')
    create_folder('Testing', 'Data/Testing')
    create_folder('yolo', 'yolo')

    data = {
        'path': '../Data',  # dataset root directory
        'train': 'Training/images',  # relative path for training images
        'val': 'Validation/images',  # relative path for validation images
        'test': 'Testing/images',  # relative path for testing images
        'names': {
            0: 'Lesion'  # class name for lesions
        }
    }

    # Write the YAML file
    
    if not os.path.exists("Data/lesion_detection.yaml"):
        with open("Data/lesion_detection.yaml", 'w') as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False)
            print(f"YAML file created at Data/lesion_detection.yaml")

    


def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Dataset installer")

    # Add the -d argument, which takes a list of dataset types
    parser.add_argument(
        '-d', '--datasets',
        type=str,
        nargs='+',  # Allows multiple values (e.g., train, val, test)
        help='Specify any datasets not to be installed (e.g., -d val test)',
        choices=['train', 'val', 'test'],  # Restrict options to 'train', 'val', 'test'
        required=False  # Makes this argument required
    )
    options=['train', 'val', 'test']
    # Parse the arguments
    args = parser.parse_args()
    if args.datasets:
        datasets_to_install = [d for d in options if d not in args.datasets]
        for i in datasets_to_install:
        # Loop through the datasets not passed by the user and install them
            install_dataset(i)
    else:
        for i in options:
            install_dataset(i)


if not os.path.exists('Data') or not os.path.exists('yolo'):
    prepare_directories()

# Instantiate the dataset with transforms
# Adjust paths for directories to raw inputs and raw masks in utils.py instead of in command line arguements.
transform = SegmentationTransform()
if os.path.exists(utils.raw_training_images_path) and os.path.exists(utils.raw_training_masks_path):
    train_dataset = ISICDataset(img_dir=utils.raw_training_images_path, mask_dir=utils.raw_training_masks_path, transform=transform)

if os.path.exists(utils.raw_validation_images_path) and os.path.exists(utils.raw_validation_masks_path):
    val_dataset = ISICDataset(img_dir=utils.raw_validation_images_path, mask_dir=utils.raw_validation_masks_path, transform=transform)

if os.path.exists(utils.raw_testing_images_path) and os.path.exists(utils.raw_testing_masks_path):
    test_dataset = ISICDataset(img_dir=utils.raw_testing_images_path, mask_dir=utils.raw_testing_masks_path, transform=transform)


if __name__ == "__main__":
    main()  
