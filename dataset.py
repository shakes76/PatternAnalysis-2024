import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms

class ISICDataset(Dataset):

    def __init__(self, img_dir='ISIC2018/ISIC2018_Task1-2_Training_Input_x2', 
                 annot_dir='ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2', 
                 mode='train', transform=None, img_size=640, grid_size=80):
        """
        Initializes the ISICDataset.

        Parameters:
            img_dir (str): Path to the directory containing images.
            annot_dir (str): Path to the directory containing annotation files (optional for test mode).
            mode (str): Mode of the dataset, either 'train' (with annotations) or 'test' (without annotations).
            transform (callable, optional): Optional transformations to apply to the images.
        """
    
        self.img_dir = img_dir
        self.annot_dir = annot_dir if mode == 'train' else None
        self.mode = mode
        self.transform = transform if transform else self.default_transforms(img_size)
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_anchors = 3
        

    def default_transforms(self, img_size):
        """
        Defines default transformations for images if none are provided.

        Returns:
            transform (callable): Transformation pipeline with resizing, normalization, and conversion to tensor.
        """
        return transforms.Compose([
            transforms.Resize((640, 640)),  # Resizing to 640x640 for consistent dimensions
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
    
    def convert_to_yolo_format(self, bbox, img_width, img_height):
        """
        Converts bounding box coordinates from (x_min, y_min, x_max, y_max) to YOLO format.

        Parameters:
            bbox (list): Bounding box in [x_min, y_min, x_max, y_max] format.
            img_width (int): Width of the image.
            img_height (int): Height of the image.

        Returns:
            tuple: Bounding box in YOLO format (x_center, y_center, width, height), normalized to the image size.
        """
        x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / 2.0 / img_width
        y_center = (y_min + y_max) / 2.0 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        return x_center, y_center, width, height

    def mask_to_bounding_boxes(self, mask):
        """
        Extracts bounding boxes from a binary mask image.

        Parameters:
            mask (PIL.Image): Grayscale mask image.

        Returns:
            list: List of bounding boxes in [x_min, y_min, x_max, y_max] format.
        """
        mask_np = np.array(mask)
        boxes = []

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y, x + w, y + h])
        
        return boxes

    def __getitem__(self, idx):
        """
        Retrieves the image (and annotation if in train mode) for a given index.

        Parameters:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: If in train mode, returns (image, yolo_boxes), else returns image.
        """
         # Load image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)  # Resize and normalize the image

        # Load and process annotations if in training mode
        if self.mode == 'train':
            # Load the segmentation mask as an image
            annot_path = os.path.join(self.annot_dir, self.img_files[idx].replace('.jpg', '_segmentation.png'))
            
            if not os.path.exists(annot_path):
                raise FileNotFoundError(f"Annotation file not found: {annot_path}")
            
            # Open the mask file as a grayscale image
            mask = Image.open(annot_path).convert("L")  # Convert to grayscale
            mask = mask.resize((640, 640))  # Resize to match image dimensions
            boxes = self.mask_to_bounding_boxes(mask)
            img_width, img_height = mask.size

            # Convert bounding boxes to YOLO format
            yolo_boxes = [self.convert_to_yolo_format(box, img_width, img_height) for box in boxes]
            
            # Prepare YOLO-style target tensor with 85 channels (for single class dataset)
            target = torch.zeros((self.num_anchors, self.grid_size, self.grid_size, 85))  # 85 for class probabilities

            for x_center, y_center, width, height in yolo_boxes:
                grid_x = int(x_center * self.grid_size)
                grid_y = int(y_center * self.grid_size)
                target[:, grid_y, grid_x, 0] = x_center
                target[:, grid_y, grid_x, 1] = y_center
                target[:, grid_y, grid_x, 2] = width
                target[:, grid_y, grid_x, 3] = height
                target[:, grid_y, grid_x, 4] = 1.0  # Object confidence score

            return image, target  # Returning the image and YOLO-style target as a tuple
        else:
            return image