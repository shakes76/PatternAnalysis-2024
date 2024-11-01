import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
import numpy as np

class ISICDataset(Dataset):
    """Custom Dataset class for YOLO model with ISIC data."""

    def __init__(self, image_dir, mask_dir, labels_path, image_size):
        self.image_size = image_size
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.labels = pd.read_csv(labels_path)

        # Load all image file names in the directory
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.samples = [self._process_sample(i) for i in range(len(self.image_files))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _process_sample(self, idx):
        """Helper function to process and return a single sample (image and target vector)."""
        # Load image and mask
        image = self._load_image(idx)
        mask = self._load_mask(idx)

        # Resize image and mask to the target size
        image = cv2.resize(image, (self.image_size, self.image_size)).astype(np.float32) / 255.0
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        # Obtain bounding box coordinates from the mask
        x, y, w, h = self._extract_bounding_box(mask)

        # Retrieve label probabilities
        label1, label2 = self.labels.iloc[idx, 1:3]
        total_prob = label1 + label2

        # Create target vector
        target_vector = np.array(
            [x + w / 2, y + h / 2, w, h, total_prob, label1, label2],
            dtype=np.float32
        )

        # Convert image to tensor format (C, H, W)
        image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        target_tensor = torch.tensor(target_vector, dtype=torch.float32)

        return image_tensor, target_tensor

    def _load_image(self, idx):
        """Loads an image given an index."""
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        return cv2.imread(img_name)

    def _load_mask(self, idx):
        """Loads the mask corresponding to the image at the given index."""
        mask_name = os.path.join(
            self.mask_dir, self.image_files[idx].replace('.jpg', '_segmentation.png')
        )
        return cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        # preparing data
    # generate CSV metadata file to read img paths and labels from it
    def generate_csv(folder, label2int):
        folder_name = os.path.basename(folder)
        labels = list(label2int)
        # generate CSV file
        df = pd.DataFrame(columns=["filepath", "label"])
        i = 0
        for label in labels:
            print("Reading", os.path.join(folder, label, "*"))
            for filepath in glob.glob(os.path.join(folder, label, "*")):
                df.loc[i] = [filepath, label2int[label]]
                i += 1
        output_file = f"{folder_name}.csv"
        print("Saving", output_file)
        df.to_csv(output_file)
    
    # generate CSV files for all data portions, labeling nevus and seborrheic keratosis
    # as 0 (benign), and melanoma as 1 (malignant)
    # you should replace "data" path to your extracted dataset path
    # don't replace if you used download_and_extract_dataset() function
    generate_csv("data/train", {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1})
    generate_csv("data/valid", {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1})
    generate_csv("data/test", {"nevus": 0, "seborrheic_keratosis": 0, "melanoma": 1})

    def _extract_bounding_box(self, mask):
        """Extracts the bounding box from the mask image."""
        _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            return x, y, w, h
        return 0, 0, 0, 0  # Return zero box if no contours are found
