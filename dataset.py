import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
import cv2

class ISICDataset(Dataset):
    def __init__(self, img_dir, annot_dir, mode='train', transform=None, img_size=640, model_output_grid_size=80):
        """
        Initializes the ISICDataset.
        """
        print("Initializing ISICDataset...")

        self.img_dir = img_dir
        self.annot_dir = annot_dir if mode == 'train' else None
        self.mode = mode
        self.img_size = img_size
        self.transform = transform if transform else self.default_transforms()
        self.num_anchors = 3
        self.grid_size = model_output_grid_size  # This should be the same as the model's output grid size (80 in your case)

        # Get list of image files
        print("Loading image files...")
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

        if self.mode == 'train':
            # Filtering only those images that have corresponding annotation files
            print("Filtering images with corresponding annotations...")
            annot_files = set([f.replace('_segmentation.png', '') for f in os.listdir(annot_dir) if f.endswith('.png')])
            self.img_files = [f for f in self.img_files if f.replace('.jpg', '') in annot_files]

        # Safeguard against empty dataset
        if not self.img_files:
            raise ValueError(f"No valid images found in {img_dir} with corresponding annotations in {annot_dir}")

        print(f"Dataset initialized with {len(self.img_files)} images.")

    def default_transforms(self):
        print("Setting default image transformations...")
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        print(f"Getting item {idx}...")
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        print(f"Loading image from: {img_path}")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # In case of error, create a dummy image to keep format consistent
            image = torch.zeros((3, self.img_size, self.img_size))

        if self.transform:
            image = self.transform(image)

        if self.mode == 'train':
            annot_filename = self.img_files[idx].replace('.jpg', '_segmentation.png')
            annot_path = os.path.join(self.annot_dir, annot_filename)
            print(f"Loading annotation from: {annot_path}")

            if not os.path.exists(annot_path):
                print("Annotation file not found, creating dummy target.")
                return image, torch.zeros((self.num_anchors, self.grid_size, self.grid_size, 85))

            try:
                mask = Image.open(annot_path).convert("L")
            except Exception as e:
                print(f"Error loading annotation {annot_path}: {e}")
                return image, torch.zeros((self.num_anchors, self.grid_size, self.grid_size, 85))

            mask = mask.resize((self.img_size, self.img_size))
            print("Annotation loaded and resized.")

            # Convert mask to numpy array and extract bounding boxes
            boxes = self.mask_to_bounding_boxes(mask)

            # Create a target tensor of size (num_anchors, grid_size, grid_size, 85) and populate it
            target_tensor = torch.zeros((self.num_anchors, self.grid_size, self.grid_size, 85))

            # Iterate over the bounding boxes and assign them to the appropriate grid cells and anchors
            img_width, img_height = mask.size
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                # Calculate grid cell positions
                grid_x = int((x_min + x_max) / 2 / img_width * self.grid_size)
                grid_y = int((y_min + y_max) / 2 / img_height * self.grid_size)

                # Ensure the grid coordinates are within bounds
                grid_x = min(max(grid_x, 0), self.grid_size - 1)
                grid_y = min(max(grid_y, 0), self.grid_size - 1)

                # Convert box to YOLO format
                x_center, y_center, width, height = self.convert_to_yolo_format(box, img_width, img_height)

                # Assign to target tensor - in this case, using the first anchor (anchor 0)
                target_tensor[0, grid_y, grid_x, 0:4] = torch.tensor([x_center, y_center, width, height])
                target_tensor[0, grid_y, grid_x, 4] = 1.0  # Objectness score
                # Set the class label - assuming one class for skin lesions
                target_tensor[0, grid_y, grid_x, 5:] = torch.zeros(80)

            return image, target_tensor
        else:
            # Return a dummy target for validation/test to ensure consistent return format
            dummy_target = torch.zeros((self.num_anchors, self.grid_size, self.grid_size, 85))
            return image, dummy_target

    def convert_to_yolo_format(self, bbox, img_width, img_height):
        x_min, y_min, x_max, y_max = bbox
        x_center = (x_min + x_max) / 2.0 / img_width
        y_center = (y_min + y_max) / 2.0 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        return x_center, y_center, width, height

    def mask_to_bounding_boxes(self, mask):
        mask_np = np.array(mask)
        boxes = []

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 0 and h > 0:  # Ensure valid bounding box
                boxes.append([x, y, x + w, y + h])

        return boxes
