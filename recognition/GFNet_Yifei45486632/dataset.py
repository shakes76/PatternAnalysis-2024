import os
from PIL import Image
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.image_paths = []
        self.labels = []
        self.label_to_index = {}
        self.transform = transform

        # Traversed to load file paths and labels
        print(f"\nLoading images from {directory}")
        for label in os.listdir(directory):
            label_folder = os.path.join(directory, label)
            
            if os.path.isdir(label_folder):
                # Map tag names to indexes
                if label not in self.label_to_index:
                    self.label_to_index[label] = len(self.label_to_index)
                
                # Get all the image files in this class
                image_files = [f for f in os.listdir(label_folder)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                print(f"Found {len(image_files)} images in {label} class")
                
                for image_file in image_files:
                    file_path = os.path.join(label_folder, image_file)
                    self.image_paths.append(file_path)
                    self.labels.append(self.label_to_index[label])

        print(f"Total images loaded: {len(self.image_paths)}")
        print("Label distribution:")
        for label, index in self.label_to_index.items():
            count = self.labels.count(index)
            print(f"{label} ({index}): {count} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load an image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    # Define the dataset directory
    directory = "./train"

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.RandomErasing(),
    ])

    # Instantiate the dataset
    dataset = CustomImageDataset(directory=directory, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Check the size of the split data set
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # Access file paths and labels
    image_paths, labels = dataset.image_paths, dataset.labels
