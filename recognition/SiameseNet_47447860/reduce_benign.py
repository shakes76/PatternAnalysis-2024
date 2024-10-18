import os
import shutil
import random


def copy_random_images(source_dir, destination_dir, percentage):
    # Make sure the source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return

    # Create the destination directory if it does not exist
    os.makedirs(destination_dir, exist_ok=True)

    # Get the list of all image files in the source directory
    # You can adjust the extensions if you have other image types (e.g., PNG, BMP)
    image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Determine how many files are 40% of the total
    num_images_to_copy = int(len(image_files) * percentage)

    # Randomly select the files to copy
    random_images = random.sample(image_files, num_images_to_copy)

    # Copy each selected file to the destination directory
    for image_file in random_images:
        src_path = os.path.join(source_dir, image_file)
        dst_path = os.path.join(destination_dir, image_file)
        shutil.copy(src_path, dst_path)

    print(f"Copied {num_images_to_copy} images from {source_dir} to {destination_dir}.")


# Define source and destination directories
source_directory = r"C:\Users\sebas\archive\original-0\0"  # Replace with the source directory path
destination_directory = r"C:\Users\sebas\archive\0"  # Replace with the destination directory path

# Call the function to copy 40% of images
copy_random_images(source_directory, destination_directory, percentage=0.0179)
