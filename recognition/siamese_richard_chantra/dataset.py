import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

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

# Define preprocessing transform for ResNet50
def preprocess_image(image):
    """Preprocess image for ResNet50 input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet50 expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    return transform(image)

# Open and preprocess image
def load_image(image_id):
    img_path = f'{img_dir}{image_id}.jpg'
    image = Image.open(img_path).convert('RGB')  # Ensure RGB format
    return preprocess_image(image)

# Train test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['target'])

# Size of splits
print(f"\nTraining set size: {len(train_data)}")
print(f"Testing set size: {len(test_data)}")

# Dataset class for DataLoader
class ImageDataset(Dataset):
    def __init__(self, data, img_dir):
        self.data = data
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = load_image(row['isic_id'])  # Now returns preprocessed tensor
        label = torch.tensor(row['target'], dtype=torch.float32)
        return image, label

# Create DataLoader
train_dataset = ImageDataset(train_data, img_dir)
test_dataset = ImageDataset(test_data, img_dir)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Test loading
if __name__ == "__main__":
    # Load and display sample image info
    sample_image_id = train_data.iloc[0]['isic_id']
    sample_tensor = load_image(sample_image_id)
    print("\nSample image tensor shape:", sample_tensor.shape)
    print("Sample image tensor range:", 
          f"min: {sample_tensor.min():.3f}, max: {sample_tensor.max():.3f}")