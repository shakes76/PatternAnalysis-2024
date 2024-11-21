"""
For loading the ADNI dataset and pre processing the data
"""

import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision.utils import make_grid


#the path to the directory on Rangpur
data_directory = '/home/groups/comp3710/ADNI/AD_NC'

#Set Hyperparameters
image_size = (256, 256) # image size (length and width)

# the mean and std values are hardcoded here, previously calculated in utils.py from the training data
transform = {
    'train': transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandAugment(num_ops=2),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),  # Convert to grayscale
        transforms.ToTensor(),
         transforms.Normalize((0.1156,), (0.2202,))
    ]),
    'test': transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(),  # Convert to grayscale
        transforms.ToTensor(),
         transforms.Normalize((0.1156,), (0.2202,))
    ]),
}

# Class to load and process the images
class ADNIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        # Use the transform specific to the mode (train, test)
        self.transform = transform 
        # Get the class names based on folder structure
        self.classes = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
        # Map class names to indices
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.image_filenames = []
        for label_dir in os.listdir(self.data_dir):
            label_path = os.path.join(self.data_dir, label_dir)
            if os.path.isdir(label_path):
                for file_name in os.listdir(label_path):
                    if file_name.endswith(('.jpeg', '.jpg')):
                        self.image_filenames.append((os.path.join(label_dir, file_name), label_dir))

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name, label = self.image_filenames[idx]
        img_path = os.path.join(self.data_dir, img_name)

       # Open the image and check all same
        image = Image.open(img_path)

        # Apply the transformation based on the mode (train, test)
        if self.transform:
            image = self.transform(image)

        # Map label to an index 
        label_idx = self.class_to_idx[label]
        
        return image, label_idx
    

# for loading and returning the train, validation and test data.
def train_dataloader(batch_size, train_size=0.8):
    print("Start DataLoading ...")
    # Create the complete dataset for training (includes validation)
    complete_train_dataset = ADNIDataset(data_dir=os.path.join(data_directory, 'train'), transform=transform['train'])

    # Split the training dataset: 80% training, 20% validation
    train_size = int(train_size * len(complete_train_dataset))
    val_size = len(complete_train_dataset) - train_size
    train_dataset, val_dataset = random_split(complete_train_dataset, [train_size, val_size])
    
    # DataLoader for batching
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    # print class names and their indices
    print(f"Classes in dataset: {complete_train_dataset.class_to_idx}")
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")

    return train_loader, val_loader

def test_dataloader(batch_size):
    # Create test data set
    test_dataset = ADNIDataset(data_dir=os.path.join(data_directory, 'test'), transform=transform['test'])
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    print(f"Classes in dataset: {test_dataset.class_to_idx}")
    print(f"Number of testing images: {len(test_dataset)}")

    # Also return the dataset for visualisation
    return test_loader, test_dataset
