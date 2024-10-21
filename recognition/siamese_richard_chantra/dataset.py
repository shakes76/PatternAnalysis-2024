import pandas as pd
from PIL import Image
import torch
from sklearn.model_selection import train_test_split

# File paths
csv_path = 'archive/train-metadata.csv'
img_dir = 'archive/train-image/image/'

# Load metadata
data = pd.read_csv(csv_path)

# Statistics
print(f"Total images: {len(data)}")
print(f"Classes distribution: \n{data['target'].value_counts()}")

# Head of metadata
print("\nFirst few rows of metadata:")
print(data.head())

# Open and return image
def load_image(image_id):
    img_path = f'{img_dir}{image_id}.jpg'
    image = Image.open(img_path)
    return image

# Train test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['target'])

# Size of splits
print(f"\nTraining set size: {len(train_data)}")
print(f"Testing set size: {len(test_data)}")

# Loading image
sample_image_id = train_data.iloc[0]['isic_id']
sample_image = load_image(sample_image_id)
sample_image.show()
