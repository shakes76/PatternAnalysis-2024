import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json  # Import the json module
import torchvision.transforms as transforms

class ISICDataset(Dataset):
    def __init__(self, img_dir='/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2', 
                 annot_dir='/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2', 
                 mode='train', transform=None):
        self.img_dir = img_dir
        self.annot_dir = annot_dir if mode == 'train' else None
        self.mode = mode
        self.transform = transform if transform else self.default_transforms()
        
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        if mode == 'train' and annot_dir is not None:
            self.annot_files = sorted([f for f in os.listdir(annot_dir) if f.endswith('_segmentation.png')]) 

    def default_transforms(self):
        return transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
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

    def __getitem__(self, idx):

        """
        Retrieves the image (and annotation if in train mode) for a given index.

        Parameters:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: If in train mode, returns (image, boxes), else returns image.
        """
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':
            annot_path = os.path.join(self.annot_dir, self.img_files[idx].replace('.jpg', '.json'))
            with open(annot_path, 'r') as f:
                annot_data = json.load(f)
            
            img_width, img_height = image.size
            boxes = [
                [obj['class_id']] + list(self.convert_to_yolo_format(obj['bbox'], img_width, img_height))
                for obj in annot_data['objects']
            ]
            boxes = torch.tensor(boxes)
            return image, boxes
        else:
            return image
