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
        Returns:
                output (torch.Tensor): shape=[b, 1], Similarity of each pair of images,
                where b = batch size
        '''
        self.path = path
        self.dataset_size = dataset_size

        self.shuffle_pairs = shuffle_pairs

        self.malignant = '1'
        self.benign = '0'

        self.feed_shape = [3, 224, 224]

        # Define transform
        self.transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), shear=0.2),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(self.feed_shape[1:], antialias=True)
        ])

        self.image_paths, self.image_classes, self.class_indices = self.load_data()
        # re-balance the classes if needed
        if augment:
            self.augment_data()
            self.image_paths, self.image_classes, self.class_indices = self.load_data()

        self.indices1, self.indices2 = self.create_pairs()

    def load_data(self):
        # load in the benign data
        image_paths = glob.glob(os.path.join(self.path, "0/*.jpg"))
        # check how much of the benign data we want to use
        if self.dataset_size and self.dataset_size <= len(image_paths):
            image_paths = random.sample(image_paths, self.dataset_size)
        # load in the malignant data
        image_paths = image_paths + (glob.glob(os.path.join(self.path, "1/*.jpg")))
        # shuffle the data together
        random.shuffle(image_paths)

        image_classes = []  # holds the class of every image (length = total num images)
        class_indices = {}  # mapping of class to every image (using the image's index) in the class

        for image_path in image_paths:
            image_class = image_path.split(os.path.sep)[-2]
            image_classes.append(image_class)

            if image_class not in class_indices:
                class_indices[image_class] = []
            class_indices[image_class].append(image_paths.index(image_path))

        return image_paths, image_classes, class_indices

    def augment_data(self):
        num_images_to_generate = len(self.class_indices[self.benign])

        # Set up the augmentation transform (without normalizing)
        augment_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), shear=0.2),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

        # Load the existing malignant images
        existing_images_count = len(self.class_indices[self.malignant])

        # Check how many more images are needed
        needed_images = num_images_to_generate - existing_images_count

        if needed_images > 0:
            print(f"Need to generate {needed_images} images.")

            # put all the used image names into a set
            used_numbers = set()
            for filename in os.listdir(os.path.join(self.path, self.malignant)):
                if filename.startswith('ISIC_') and filename.endswith('.jpg'):
                    # Extract the 7-digit number from the filename
                    number = filename.split('_')[1].split('.')[0]
                    used_numbers.add(number)

            # Loop to generate additional images
            for i in range(needed_images):
                # Randomly choose an existing image to augment
                img_index = np.random.choice(self.class_indices[self.malignant])
                img_path = self.image_paths[img_index]
                img = Image.open(img_path)

                # Apply augmentation transform to the image
                augmented_img = augment_transform(img)

                # Save the augmented image back to the same directory with a new name
                while True:
                    # Generate a random 7-digit number
                    random_number = '{:07d}'.format(random.randint(0, 9999999))

                    if random_number not in used_numbers:
                        # If the number is not used, return the new unique name
                        img_name = f'ISIC_{random_number}.jpg'
                        break

                augmented_img.save(os.path.join(self.path, self.malignant, img_name))
            print("Done Augmenting")
        else:
            print(f"No new images are needed. The directory already has {existing_images_count} images.")

    def create_pairs(self):
        '''
        Creates two lists of indices that will form the pairs, to be fed for training or evaluation.
        '''
        # indices1 is the index for number of total images there are (0, 1, ..., n)
        indices1 = np.arange(len(self.image_paths))

        if self.shuffle_pairs:
            np.random.seed(int(time.time()))
            np.random.shuffle(indices1)
        else:
            # If shuffling is set to off, set the random seed to 1, to make it deterministic.
            np.random.seed(1)

        # make an array of random true/false values with length = the total number of images
        select_pos_pair = np.random.rand(len(self.image_paths)) < 0.5

        indices2 = []

        # iterate through every image index and its corresponding random true/false value
        for i, pos in zip(indices1, select_pos_pair):
            class1 = self.image_classes[i]
            if pos:
                # we want a matching image, make classes the same
                class2 = class1
            else:
                # else we want a different class of image, randomly pick other class after removing class 1
                class2 = np.random.choice(list(set(self.class_indices.keys()) - {class1}))
            # randomly pick an image from this class
            idx2 = np.random.choice(self.class_indices[class2])
            # add this image to the indices2, its corresponding pair and whether they match
            # are saved in image_paths and select_pos_pair
            indices2.append(idx2)

        # vectorise so we can turn it to a tensor
        return indices1, np.array(indices2)

    def __iter__(self):
        '''
                    Iterates through the dataset split and yields pairs of images and similarity labels.
                    '''
        for idx, idx2 in zip(self.indices1, self.indices2):
            image_path1 = self.image_paths[idx]
            image_path2 = self.image_paths[idx2]

            class1 = self.image_classes[idx]
            class2 = self.image_classes[idx2]

            image1 = Image.open(image_path1).convert("RGB")
            image2 = Image.open(image_path2).convert("RGB")

            if self.transform:
                image1 = self.transform(image1).float()
                image2 = self.transform(image2).float()

            yield (image1, image2), torch.FloatTensor([class1 == class2]), (class1, class2)

    def __len__(self):
        return len(self.image_paths)
