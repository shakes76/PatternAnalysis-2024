import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
## On val 2152 imgs:    Mean: 0.12288379669189453, Std: 0.2244586944580078
## On train 19368 imgs: Mean: 0.19868804514408112, Std: 0.24770835041999817
import argparse
from utils import get_args_parser


class ADNI_Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (string): Root directory of the dataset, e.g. 'data/'.
            split (string): 'train', 'val', or 'test' to specify which data to load.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        
        # Classes NC and AD
        self.classes = ['NC', 'AD']
        self.images = []
        self.labels = []

        # Populate the list of images and labels
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.png') or img_name.endswith('.jpg') or img_name.endswith('.jpeg'):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)  # 0 for NC, 1 for AD
        print('ADNI Dataset with {} instances for {} phase'.format(len(self.images), split))

    def __len__(self):
        """Returns the total number of samples"""
        return len(self.images)

    def __getitem__(self, idx):
        """Generates one sample of data"""
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('L')  # Grayscale ('L' mode)

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        return image, label



def build_dataset(is_train, args, infer_no_resize=False):
    transform = build_transform(is_train, args, infer_no_resize)

    if args.data_set == 'ADNI':
        dataset = ADNI_Dataset('data/', 'train' if is_train else 'val', transform=transform)
        nb_classes = 2

    return dataset, nb_classes


def build_transform(is_train, args, infer_no_resize=False):
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(sharpness_factor=0.9, p=0.1),
                transforms.Grayscale(num_output_channels=1),  # Converts to grayscale
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=0.19868804514408112,
                    std=0.24770835041999817,
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=1),  # Converts to grayscale
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=0.12288379669189453,
                    std=0.2244586944580078,
                ),
            ]
        )
    return transform



def get_mean_and_std(dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        # Flatten the images into a single dimension per batch
        images = images.view(images.size(0), -1)  # (batch_size, width*height)
        # Compute the sum and square sum for each batch
        batch_samples = images.size(0)  # Number of images in the current batch
        total_images_count += batch_samples
        mean += images.mean(1).sum()
        std += images.std(1).sum() 
    # Divide by total number of images to get the mean
    mean /= total_images_count
    std /= total_images_count
    return mean.item(), std.item()



if __name__ == "__main__":  # Add this guard to prevent multiprocessing issues
    parser = argparse.ArgumentParser('GFNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    ## print(args)
    dataset_val, nb_classes = build_dataset(is_train=False, args=args)
    # Assuming dataset_val is already defined
    mean, std = get_mean_and_std(dataset_val)
    print(f"Mean: {mean}, Std: {std}")