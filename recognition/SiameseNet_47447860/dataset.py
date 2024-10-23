import os
import glob
import time
import numpy as np
from PIL import Image
import random
import torch
from torchvision import transforms

class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, path, shuffle_pairs=True, dataset_size=None, augment=False):
        '''
        Initialize the dataset class.

        Args:
            path (str): Path to the dataset directory.
            shuffle_pairs (bool): Whether to shuffle pairs of images.
            dataset_size (int, optional): Number of samples to use from the dataset.
            augment (bool): Whether to perform data augmentation.
        '''
        self.path = path
        self.dataset_size = dataset_size
        self.shuffle_pairs = shuffle_pairs
        self.malignant = '1'  # Class label for malignant images
        self.benign = '0'  # Class label for benign images
        self.feed_shape = [3, 224, 224]  # Image dimensions (channels, height, width)

        # Define transformation pipeline for data augmentation
        self.transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), shear=0.2),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(self.feed_shape[1:], antialias=True)
        ])

        # Load dataset information
        self.image_paths, self.image_classes, self.class_indices = self.load_data()
        # Perform data augmentation if required
        if augment:
            self.augment_data()
            self.image_paths, self.image_classes, self.class_indices = self.load_data()

        # Create pairs of image indices for training
        self.indices1, self.indices2 = self.create_pairs()

    def load_data(self):
        '''
        Load image paths and classes from the dataset.

        Returns:
            tuple: Lists of image paths, image classes, and class indices.
        '''
        # Load benign image paths
        image_paths = glob.glob(os.path.join(self.path, "0/*.jpg"))
        # Select a subset of benign data if dataset_size is specified
        if self.dataset_size and self.dataset_size <= len(image_paths):
            image_paths = random.sample(image_paths, self.dataset_size)
        # Load malignant image paths and combine with benign images
        image_paths = image_paths + (glob.glob(os.path.join(self.path, "1/*.jpg")))
        # Shuffle all images
        random.shuffle(image_paths)

        image_classes = []  # Holds the class of every image
        class_indices = {}  # Maps each class to indices of images belonging to that class

        for image_path in image_paths:
            image_class = image_path.split(os.path.sep)[-2]
            image_classes.append(image_class)

            if image_class not in class_indices:
                class_indices[image_class] = []
            class_indices[image_class].append(image_paths.index(image_path))

        return image_paths, image_classes, class_indices

    def augment_data(self):
        '''
        Perform data augmentation on malignant images to balance class distribution.
        '''
        num_images_to_generate = len(self.class_indices[self.benign])

        # Set up the augmentation transform (without normalizing)
        augment_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), shear=0.2),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

        # Load the existing malignant images
        existing_images_count = len(self.class_indices[self.malignant])

        # Check how many more images are needed to balance the dataset
        needed_images = num_images_to_generate - existing_images_count

        if needed_images > 0:
            print(f"Need to generate {needed_images} images.")

            # Track used image names to avoid duplicates
            used_numbers = set()
            for filename in os.listdir(os.path.join(self.path, self.malignant)):
                if filename.startswith('ISIC_') and filename.endswith('.jpg'):
                    # Extract the 7-digit number from the filename
                    number = filename.split('_')[1].split('.')[0]
                    used_numbers.add(number)

            # Generate additional augmented images
            for i in range(needed_images):
                # Randomly choose an existing image to augment
                img_index = np.random.choice(self.class_indices[self.malignant])
                img_path = self.image_paths[img_index]
                img = Image.open(img_path)

                # Apply augmentation transform to the image
                augmented_img = augment_transform(img)

                # Generate a unique filename for the augmented image
                while True:
                    # Generate a random 7-digit number
                    random_number = '{:07d}'.format(random.randint(0, 9999999))

                    if random_number not in used_numbers:
                        # If the number is not used, create the new unique name
                        img_name = f'ISIC_{random_number}.jpg'
                        break

                # Save the augmented image to the malignant directory
                augmented_img.save(os.path.join(self.path, self.malignant, img_name))
            print("Done Augmenting")
        else:
            print(f"No new images are needed. The directory already has {existing_images_count} images.")

    def create_pairs(self):
        '''
        Creates two lists of indices that will form the pairs, to be fed for training or evaluation.

        Returns:
            tuple: Two lists of indices representing pairs of images.
        '''
        # indices1 is the list of indices for all images
        indices1 = np.arange(len(self.image_paths))

        # Shuffle pairs if shuffle_pairs is True
        if self.shuffle_pairs:
            np.random.seed(int(time.time()))
            np.random.shuffle(indices1)
        else:
            # Set the seed to make shuffling deterministic
            np.random.seed(1)

        # Randomly select if pairs should have matching classes or not
        select_pos_pair = np.random.rand(len(self.image_paths)) < 0.5

        indices2 = []

        # Iterate through image indices and generate pairs
        for i, pos in zip(indices1, select_pos_pair):
            class1 = self.image_classes[i]
            if pos:
                # If true, select another image from the same class
                class2 = class1
            else:
                # Else, select an image from a different class
                class2 = np.random.choice(list(set(self.class_indices.keys()) - {class1}))
            # Randomly pick an image from the selected class
            idx2 = np.random.choice(self.class_indices[class2])
            indices2.append(idx2)

        # Return the pair indices as numpy arrays
        return indices1, np.array(indices2)

    def __iter__(self):
        '''
        Iterate through the dataset and yield pairs of images along with similarity labels.

        Yields:
            tuple: A tuple containing a pair of images, similarity label, and their respective classes.
        '''
        for idx, idx2 in zip(self.indices1, self.indices2):
            image_path1 = self.image_paths[idx]
            image_path2 = self.image_paths[idx2]

            class1 = self.image_classes[idx]
            class2 = self.image_classes[idx2]

            # Load and convert images to RGB
            image1 = Image.open(image_path1).convert("RGB")
            image2 = Image.open(image_path2).convert("RGB")

            # Apply transformations if available
            if self.transform:
                image1 = self.transform(image1).float()
                image2 = self.transform(image2).float()

            # Yield the image pair, similarity label, and class information
            yield (image1, image2), torch.FloatTensor([class1 == class2]), (class1, class2)

    def __len__(self):
        '''
        Return the total number of images in the dataset.

        Returns:
            int: The total number of images.
        '''
        return len(self.image_paths)
