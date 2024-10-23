from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import os
from PIL import Image
import matplotlib.pyplot as plt
from utils import *

# server path
train_path_server = '/home/groups/comp3710/ADNI/AD_NC/train/'
test_path_server = '/home/groups/comp3710/ADNI/AD_NC/test/'

class ADNI_Dataset(Dataset):
    """
    Custom Dataset loader for loading ADNI brain data.

    Args:
        root_dir : Path to the root directory containing image subfolders ('NC' and 'AD').
        transform : A function/transform that takes in a PIL image and returns a transformed version.

    Attributes:
        root_dir : Path to the root directory.
        transform : Image transformation function.
        image_paths : List of full paths to images.
        labels : List of labels corresponding to the images (0 for NC, 1 for AD).
        category_counts : Dictionary counting the number of images in 'NC' and 'AD' categories.
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.category_counts = {'NC': 0, 'AD': 0}  

        for label, sub_dir in enumerate(['NC', 'AD']):
            sub_dir_path = os.path.join(root_dir, sub_dir)
            num_images = len(os.listdir(sub_dir_path))  
            self.category_counts[sub_dir] = num_images  

            for image_name in os.listdir(sub_dir_path):
                self.image_paths.append(os.path.join(sub_dir_path, image_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_category_counts(self):
        return self.category_counts

def preprocess_image(image_path):
    """
    Function to preprocess image to be predicted

    Args:
        image_path: String of the image path
    
    Returns:
        image: Preprocessed image tensor suitable for input to the model.
    """
    image = Image.open(image_path).convert('RGB')
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1155, 0.1155, 0.1155], std=[0.2224, 0.2224, 0.2224])
    ])
    image = transform(image) 
    image = image.unsqueeze(0)
    return image

# Define transformation (data augmentation) for the training data
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1155, 0.1155, 0.1155], std=[0.2224, 0.2224, 0.2224])
    ])

# Define transformation (data augmentation) for the test data
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1155, 0.1155, 0.1155], std=[0.2224, 0.2224, 0.2224])
])

transform_no_augmentation = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1155, 0.1155, 0.1155], std=[0.2224, 0.2224, 0.2224])
])

train_dataset = ADNI_Dataset(train_path_server, transform=transform)
test_dataset = ADNI_Dataset(test_path_server, transform=transform_test)

# Size of validation data 
val_split = 0.2 

train_size = int((1 - val_split) * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# Define train, validation, and test data loader
train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

classes = ['NC', 'AD']

# save sample of train images augmented
show_sample_data(train_loader, classes, 'img_training_samples_augmented')

# save sample of test images augmented
# shuffle=True for the purpose of showing the samples
test_loader_augmented = DataLoader(test_dataset, batch_size=16, shuffle=True)
show_sample_data(test_loader_augmented, classes, 'img_testing_samples_augmented')


# show sample of original dataset
train_dataset_no_augmented = ADNI_Dataset(train_path_server, transform=transform_no_augmentation)
test_dataset_no_augmented = ADNI_Dataset(test_path_server, transform=transform_no_augmentation)

train_loader_no_augmented = DataLoader(train_dataset_no_augmented, batch_size=16, shuffle=True)
# shuffle=True for the purpose of showing the samples
test_loader_no_augmented = DataLoader(test_dataset_no_augmented, batch_size=16, shuffle=True)

# save sample of training images no augmented
show_sample_data(train_loader_no_augmented, classes, 'img_training_samples_original')
show_sample_data(test_loader_no_augmented, classes, 'img_testing_samples_original')

# get counts for the training dataset
train_counts = train_dataset.get_category_counts()
print(f"Total training data (before splitting): {train_counts}")

# get counts for the train subset after splitting
train_subset_counts = {'NC': 0, 'AD': 0}
for _, label in train_subset:
    if label == 0:
        train_subset_counts['NC'] += 1
    else:
        train_subset_counts['AD'] += 1
print(f"Training subset counts: {train_subset_counts}")

# get counts for the validation subset
val_subset_counts = {'NC': 0, 'AD': 0}
for _, label in val_subset:
    if label == 0:
        val_subset_counts['NC'] += 1
    else:
        val_subset_counts['AD'] += 1
print(f"Validation subset counts: {val_subset_counts}")

# get counts for the test dataset
test_counts = test_dataset.get_category_counts()
print(f"Test dataset counts: {test_counts}")