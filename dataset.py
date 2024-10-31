import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ISICDataset(Dataset):
    def __init__(self, img_dir='/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2', 
                 annot_dir='/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2', 
                 mode='train', transform=None):
        """
        Initializes the ISICDataset.

        Parameters:
            img_dir (str): Path to the directory containing images.
            annot_dir (str): Path to the directory containing annotation files (optional for test mode).
            mode (str): Mode of the dataset, either 'train' (with annotations) or 'test' (without annotations).
            transform (callable, optional): Optional transformations to apply to the images.
        """
        self.img_dir = img_dir
        self.annot_dir = annot_dir if mode == 'train' else None  # Set to None in test mode
        self.mode = mode
        self.transform = transform if transform else self.default_transforms()
        
        # Get a list of all image files in the directory
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

        # If in training mode, load the annotation files
        if mode == 'train' and annot_dir is not None:
            self.annot_files = sorted([f for f in os.listdir(annot_dir) if f.endswith('_segmentation.png')]) 

    def default_transforms(self):
        """
        Defines default transformations for images if none are provided.

        Returns:
            transform (callable): Transformation pipeline with resizing, normalization, and conversion to tensor.
        """
        return transforms.Compose([
            transforms.Resize((640, 640)),  # Resizing to 640x640 for YOLO compatibility
            transforms.ToTensor(),  # Converting image to PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalizing with ImageNet standard 
        ])

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.img_files)

    def __getitem__(self, idx):
        """
        Retrieves the image (and annotation if in train mode) for a given index.

        Parameters:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: If in train mode, returns (image, mask), else returns image.
        """
        # Loading images from file
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")  # Ensure image is in RGB format

        # If in training mode, load annotations
        if self.mode == 'train' and self.annot_dir is not None:
            # Load corresponding annotation (segmentation mask) file
            mask_path = os.path.join(self.annot_dir, self.img_files[idx].replace('.jpg', '_segmentation.png'))
            mask = Image.open(mask_path).convert("L")  # Load mask as grayscale
            
            # Apply transformations to the image and mask
            if self.transform:
                image = self.transform(image)
                mask = transforms.ToTensor()(mask)  # Convert mask to tensor (binary mask)

            # Return both the transformed image and mask
            return image, mask
        else:
            # If in test mode, only return the transformed image (no annotations)
            if self.transform:
                image = self.transform(image)
                
            return image
