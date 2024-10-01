import os
from PIL import Image

# Directories for datasets
train_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'
test_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'
validate_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'
save_dir = 'saved_images'


# Load a few images and print their shape
def check_image_shape(directory, num_images=5):
    img_paths = [os.path.join(directory, img) for img in os.listdir(directory)]
    for img_path in img_paths[:num_images]:
        image = Image.open(img_path)
        print(f"Image: {img_path}, Shape: {image.size}")

# Check shapes for the first few images in the train directory
check_image_shape(train_dir)