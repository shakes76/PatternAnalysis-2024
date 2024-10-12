import os
import shutil
import pandas as pd

# Paths
base_dir = '~/.kaggle'
# base_dir = r'C:\Users\sebas\archive'
csv_file = os.path.join(base_dir, 'train-metadata.csv')
image_dir = os.path.join(base_dir, 'train-image', 'image')

# Output directories for classification
classification0_dir = os.path.join(base_dir, '0')
classification1_dir = os.path.join(base_dir, '1')

# Create the classification directories if they don't exist
os.makedirs(classification0_dir, exist_ok=True)
os.makedirs(classification1_dir, exist_ok=True)

# Read the CSV file
metadata = pd.read_csv(csv_file)

# Assuming the CSV has columns 'image_name' and 'classification'
# Adjust these column names according to your actual CSV structure
for _, row in metadata.iterrows():
    image_name = row['isic_id']
    classification = row['target']

    # Add .jpg extension to the image name
    img_filename = f"{image_name}.jpg"

    # Source image path
    src_path = os.path.join(image_dir, img_filename)

    # Destination path based on classification
    if classification == 0:  # Classification 0 - Benign
        dest_path = os.path.join(classification0_dir, img_filename)
    elif classification == 1:  # Classification 1 - Malignant
        dest_path = os.path.join(classification1_dir, img_filename)
    else:
        print(f"Unknown classification '{classification}' for image '{image_name}'")
        continue

    # Move or copy the image to the appropriate directory
    shutil.copy(src_path, dest_path)  # You can use shutil.move if you prefer to move instead of copy

print("Images have been classified and moved successfully.")
