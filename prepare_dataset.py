import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

def move_dataset():
    # Source directories of data
    input_dir = '/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2'
    mask_dir = '/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2'

    # Create directories to move data to
    for dir_path in ['data/images/train', 'data/images/val', 'data/masks/train', 'data/masks/val']:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Get list of image files (.jpg)
    image_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            image_files.append(filename)

    # Split the dataset into train and validation sets (80-20 split)
    train_files, val_files = train_test_split(
        image_files,
        test_size=0.2,
        random_state=40
    )

    # Helper used to get mask's file name from the image's name
    def get_mask_filename(image_filename):
        return image_filename.replace('.jpg', '_segmentation.png')

    # Copy training files
    for filename in train_files:
        # Copy input image
        shutil.copy2(
            os.path.join(input_dir, filename),
            os.path.join('data/images/train', filename)
        )
        # Copy mask
        mask_filename = get_mask_filename(filename)
        shutil.copy2(
            os.path.join(mask_dir, mask_filename),
            os.path.join('data/masks/train', mask_filename)
        )

    # Copy validation files
    for filename in val_files:
        # Copy input image
        shutil.copy2(
            os.path.join(input_dir, filename),
            os.path.join('data/images/val', filename)
        )
        # Copy mask
        mask_filename = get_mask_filename(filename)
        shutil.copy2(
            os.path.join(mask_dir, mask_filename),
            os.path.join('data/masks/val', mask_filename)
        )

move_dataset()