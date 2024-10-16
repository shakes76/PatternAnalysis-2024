"""
Got inspiration from datasets.py file of the following github repo:
https://github.com/shakes76/GFNet
and my demo 2 brain gan data-loading code
"""

import torchvision.transforms as tf
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class ADNIDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        self.images, self.labels = self.load_data()
    
    def load_data(self):
        images = []
        labels = []

        sub_directory = "train" if self.train else "test"
        label_names = {"AD": 1, "NC": 0}

        for label in label_names.keys():
            label_directory = os.path.join(self.root_dir, sub_directory, label)
            for img_name in os.listdir(label_directory):
                img_path = os.path.join(label_directory, img_name)
                images.append(img_path)
                labels.append(label_names[label])
        
        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        image = Image.open(img_path).convert('L')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloaders(data_dir, batch_size=32, crop_size=224, image_size=224):
    if data_dir is None:
        # Rangpur
        data_dir = "/home/groups/comp3710/ADNI/AD_NC"
        # Local
        # data_dir = "../../resources/AD_NC"
    train_transform = tf.Compose([
        tf.Grayscale(num_output_channels=1),
        tf.CenterCrop(crop_size),
        tf.Resize((image_size, image_size)),
        tf.ToTensor(),
        # Hardcoded to value found below
        tf.Normalize(mean=[0.1415],
                     std=[0.2385]),
        # tf.Normalize(mean=[0.1156],
        #              std=[0.2198],),
        tf.RandomHorizontalFlip(),
        tf.RandomVerticalFlip(),
    ])
    test_transform = tf.Compose([
        tf.Grayscale(num_output_channels=1),
        tf.CenterCrop(crop_size),
        tf.Resize((image_size, image_size)),
        tf.ToTensor(),
        # Hardcoded to value found below
        tf.Normalize(mean=[0.1415],
                     std=[0.2385]),
        # tf.Normalize(mean=[0.1156],
        #              std=[0.2198],),
    ])
    
    # Create datasets
    train_dataset = ADNIDataset(root_dir=data_dir, transform=train_transform, train=True)
    test_dataset = ADNIDataset(root_dir=data_dir, transform=test_transform, train=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

# Running the function below, get_mean_std(), prints:
# tensor(0.1415) tensor(0.2385), so mean = 0.1415, std = 0.2385.
# This is where the "magic values" in the above section come from.
# Inspired by Yuvraj Fowdar's edstem post: https://edstem.org/au/courses/18266/discussion/2253272.

# Other values for potential future reference (above is with centercrop and resize):
# No centercrop and no resize: tensor(0.1155) tensor(0.2224) 
# No centercrop but resize: tensor(0.1156) tensor(0.2198) 
# Centercrop and no resize: tensor(0.1415) tensor(0.2385) 

# def get_mean_std():

#     # data_dir = "/home/groups/comp3710/ADNI/AD_NC"
#     data_dir = "/home/daqoblade/Uni/2024 Sem 2/COMP3710/PatternAnalysis-2024/resources/AD_NC"
#     batch_size = 32

#     transform = tf.Compose([
#         tf.Grayscale(num_output_channels=1),
#         tf.CenterCrop(224),
#         tf.Resize((224, 224)),
#         tf.ToTensor(),
#     ])
    
#     train_dataset = ADNIDataset(root_dir=data_dir, transform=transform, train=True)
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

#     num_images = 0
#     mean = 0.0
#     std = 0.0
#     for images, _ in train_loader:
#         batch_size = images.size(0)
#         images = images.view(batch_size, -1)
#         mean += images.mean(1).sum(0)
#         std += images.std(1).sum(0)
#         num_images += batch_size
    
#     mean /= num_images
#     std /= num_images

#     print(mean, std)

# if __name__ == '__main__':
#     get_mean_std()