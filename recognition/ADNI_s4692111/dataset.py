import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ADNI_Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
            image_dir (str): the path that store the .jpeg picture.
            labels (list): to distinguish between NC and AD, NC is 0, AD is 1.
            transform (callable, optional): the optional for dataset, the default is None.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # return the size of data
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load the picture and label by idx.
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')  # change jpeg to greyscale.

        # If has optional, use it.
        if self.transform:
            img = self.transform(img)

        # get the label
        label = self.labels[idx]

        return img, label


def load_images_from_folder(image_dir):
    """
    get all the things in "train" and "test" folder
    Returns:
        datasets (dict): a dictionary contain 'train' and 'test', 
        which include every picture path and label.
    """
    datasets = {'train': {'image_paths': [], 'labels': []}, 'test': {'image_paths': [], 'labels': []}}

    # get file in 'train' and 'test'
    for phase in ['train', 'test']:
        phase_dir = os.path.join(image_dir, phase)
        
        # get file in 'AD' and 'NC'
        for label_type in ['AD', 'NC']:
            label_dir = os.path.join(phase_dir, label_type)
            label = 1 if label_type == 'AD' else 0

            # apend all pictures in folder
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                datasets[phase]['image_paths'].append(img_path)
                datasets[phase]['labels'].append(label)
    
    return datasets

def get_data_loaders(image_dir, batch_size=8, shuffle=True):
    # define the transform options
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # To model size.
        transforms.ToTensor(),  # To tensor
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    datasets = load_images_from_folder(image_dir)

    train_dataset = ADNI_Dataset(datasets['train']['image_paths'], datasets['train']['labels'], transform=transform)
    test_dataset = ADNI_Dataset(datasets['test']['image_paths'], datasets['test']['labels'], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
    
