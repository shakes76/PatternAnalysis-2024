import albumentations as A  # Correct import
import os
from PIL import Image
import numpy as np
import cv2

# reformat the directory structure
original_paths = ['data/train/AD', 'data/train/NC', 'data/test/AD', 'data/test/NC']
new_paths = ['train/AD', 'train/NC', 'test/AD', 'test/NC']

for original_path, new_path in zip(original_paths, new_paths):
    os.makedirs(new_path, exist_ok=True)
    for image_file in os.listdir(original_path):
        image_path = os.path.join(original_path, image_file)
        new_image_path = os.path.join(new_path, image_file)
        os.rename(image_path, new_image_path)

# Define your input directories
input_paths = ['train/AD', 'train/NC', 'test/AD', 'test/NC']

# Define the resize transformation (no normalization)
transform = A.Compose([
    A.PadIfNeeded(min_height=256, min_width=256, value=1, border_mode=cv2.BORDER_CONSTANT, always_apply=True),
    A.Resize(256, 256),  # Only resize
])

# Loop over each input path and process the images
for input_dir in input_paths:
    for image_file in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_file)  # Full path to the image
        
        # Open and convert the image to a NumPy array
        image = np.array(Image.open(image_path))
        
        # Apply resize transformation
        resized_image = transform(image=image)['image']
        
        # Convert the resized NumPy array back to an image
        resized_image_pil = Image.fromarray(resized_image)
        
        
        # Save the resized image back to the same directory, overwriting the original
        resized_image_pil.save(image_path)

