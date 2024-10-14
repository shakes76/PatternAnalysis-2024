import os
import glob
import time

import numpy as np
from PIL import Image

import torch
from torchvision import transforms


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, path, augment, shuffle_pairs=True):
        '''
        Returns:
                output (torch.Tensor): shape=[b, 1], Similarity of each pair of images,
                where b = batch size
        '''
        self.path = path
        self.augment = augment

        self.shuffle_pairs = shuffle_pairs

        self.train_portion = 0.7
        self.train_val_index = 0

        self.val_portion = 0.15
        self.val_test_index = 0

        self.test_portion = 0.15

        self.malignant = '1'
        self.benign = '0'

        # self.augment_malignant()

        self.image_paths, self.image_classes, self.class_indices = self.load_data()
        self.indices1, self.indices2 = self.create_pairs()

    # def augment_malignant(self):
        # if balancing doesn't work well, then we can augment to give it more training data

    def load_data(self):
        image_paths = glob.glob(os.path.join(self.path, "*/*.jpg"))
        image_classes = []  # holds the class of every image (length = total num images)
        class_indices = {}  # mapping of class to every image in the class

        for image_path in image_paths:
            image_class = image_path.split(os.path.sep)[-2]
            image_classes.append(image_class)

            if image_class not in class_indices:
                class_indices[image_class] = []
            class_indices[image_class].append(image_paths.index(image_path))

        return image_paths, image_classes, class_indices

    def create_pairs(self):
        '''
        Creates two lists of indices that will form the pairs, to be fed for training or evaluation.
        '''
        # we want to use an indices1 that balances malignant and benign (we know malignant is much less)
        # self.indices1 = np.arange(2 * len(self.class_indices[self.malignant]))

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

        # find the train/val/test split index
        self.train_val_index = int(len(self.image_paths) * self.train_portion)
        self.val_test_index = int(len(self.image_paths) * (self.train_portion + self.val_portion))

        # vectorise so we can turn it to a tensor
        return indices1, np.array(indices2)

    def get_split(self, mode):
        '''
        Create and return the split dataset for the given mode (train, val, or test).
        '''
        if mode == 'train':
            return self.Split(self.image_paths, self.image_classes, self.indices1[:self.train_val_index],
                              self.indices2[:self.train_val_index], self.augment)
        elif mode == 'val':
            return self.Split(self.image_paths, self.image_classes,
                              self.indices1[self.train_val_index:self.val_test_index],
                              self.indices2[self.train_val_index:self.val_test_index], self.augment)
        elif mode == 'test':
            return self.Split(self.image_paths, self.image_classes, self.indices1[self.val_test_index:],
                              self.indices2[self.val_test_index:], self.augment)
        else:
            raise ValueError(f"Unknown mode {mode}. Must be one of: 'train', 'val', 'test'.")

    class Split(torch.utils.data.IterableDataset):
        '''
        Split class to handle iteration over a specific subset of the data (train, val, or test).
        '''

        def __init__(self, image_paths, image_classes, indices1, indices2, augment):
            self.image_paths = image_paths
            self.image_classes = image_classes
            self.indices1 = indices1
            self.indices2 = indices2
            self.augment = augment
            self.feed_shape = [3, 224, 224]
            #self.feed_shape = [3, 256, 256]

            # Define transforms
            if self.augment:
                # If images are to be augmented, add extra operations for it (first two).
                self.transform = transforms.Compose([
                    transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.Resize(self.feed_shape[1:], antialias=True)
                ])
            else:
                # If no augmentation is needed then apply only the normalization and resizing operations.
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.Resize(self.feed_shape[1:], antialias=True)
                ])

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
            return len(self.indices1)

    def __len__(self):
        return len(self.image_paths)