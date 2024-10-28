import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import Counter

class ADNIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def load_train_val_datasets(folder_path, transform=None, val_split=0.2, random_state=42):
    # Get paths of AD and NC directories
    ad_path = os.path.join(folder_path, "AD")
    nc_path = os.path.join(folder_path, "NC")

    # Create lists for image paths and corresponding labels
    image_paths = []
    labels = []

    for label, class_dir in enumerate([ad_path, nc_path]):
        for img_name in os.listdir(class_dir):
            if img_name.endswith(".jpeg"):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(label)  # Set class AD as 0, class NC as 1

    # Split data into training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=val_split, stratify=labels, random_state=random_state
    )

    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)

    print(f"Training set - AD: {train_counter[0]}, NC: {train_counter[1]}")
    print(f"Validation set - AD: {val_counter[0]}, NC: {val_counter[1]}")

    # Create dataset objects
    train_dataset = ADNIDataset(train_paths, train_labels, transform=transform)
    val_dataset = ADNIDataset(val_paths, val_labels, transform=transform)

    return train_dataset, val_dataset

# Define the transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load train and validation datasets
train_dataset, val_dataset = load_train_val_datasets("./ADNI/AD_NC/train", transform=transform)

# Load train and validation data
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Load test dataset
def load_test_dataset(folder_path, transform=None):
    test_image_paths = []
    test_labels = []

    # Get paths of AD and NC directories
    ad_path = os.path.join(folder_path, "AD")
    nc_path = os.path.join(folder_path, "NC")

    # Collect all test images and labels
    for label, class_dir in enumerate([ad_path, nc_path]):
        for img_name in os.listdir(class_dir):
            if img_name.endswith(".jpeg"):
                test_image_paths.append(os.path.join(class_dir, img_name))
                test_labels.append(label)

    # Create the test dataset
    return ADNIDataset(test_image_paths, test_labels, transform=transform)

# Load the test dataset and create DataLoader
test_dataset = load_test_dataset("./ADNI/AD_NC/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

