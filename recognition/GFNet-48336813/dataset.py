import os
import argparse
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import get_args_parser

ADNI_DEFAULT_MEAN_TRAIN = 0.19868804514408112
ADNI_DEFAULT_STD_TRAIN = 0.24770835041999817

ADNI_DEFAULT_MEAN_VAL = 0.12288379669189453
ADNI_DEFAULT_STD_VAL = 0.2244586944580078

ADNI_DEFAULT_MEAN_TEST = 0.12404339015483856
ADNI_DEFAULT_STD_TEST = 0.2250228226184845

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


def build_dataset(split, args):
    transform = build_transform(split, args)
    if args.data_set == 'ADNI':
        dataset = ADNI_Dataset('data/', split=split, transform=transform)
        nb_classes = 2
    else: NotImplementedError
    return dataset, nb_classes


def build_transform(split, args):
    if split == "train":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(sharpness_factor=0.9, p=0.1),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=ADNI_DEFAULT_MEAN_TRAIN,
                    std=ADNI_DEFAULT_STD_TRAIN,
                ),
            ]
        )
    elif split == "val":
        transform = transforms.Compose(
            [
                transforms.Resize(args.input_size),
                transforms.CenterCrop(args.input_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=ADNI_DEFAULT_MEAN_VAL,
                    std=ADNI_DEFAULT_STD_VAL,
                ),
            ]
        )
    elif split == "test":
        transform = transforms.Compose(
            [
                transforms.Resize(args.input_size),
                transforms.CenterCrop(args.input_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=ADNI_DEFAULT_MEAN_TEST,
                    std=ADNI_DEFAULT_STD_TEST,
                ),
            ]
        )
    else: NotImplementedError
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser('GFNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # Get dataset mean and std for required split
    split = 'test'  # Change to 'train', 'val', or 'test'
    dataset, nb_classes = build_dataset(split=split, args=args)
    mean, std = get_mean_and_std(dataset)
    print(f"Mean: {mean}, Std: {std}")