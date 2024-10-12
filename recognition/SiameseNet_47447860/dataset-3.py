import os
import glob
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, path, shuffle_pairs=True):
        '''
        Returns:
                output (torch.Tensor): shape=[b, 1], Similarity of each pair of images,
                where b = batch size
        '''
        self.path = path

        self.feed_shape = [3, 256, 256]
        self.shuffle_pairs = shuffle_pairs

        self.train_portion = 0.7
        self.train_val_index = 0

        self.val_portion = 0.15
        self.val_test_index = 0

        self.test_portion = 0.15

        self.malignant = '1'
        self.benign = '0'

        self.augment_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(self.feed_shape[1:])
        ])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(self.feed_shape[1:])
        ])

        self.augment_malignant()

        self.create_pairs()

    # def augment_malignant(self):
        # if balancing doesn't work well, then we can augment to give it more training data

    def create_pairs(self):
        '''
        Creates two lists of indices that will form the pairs, to be fed for training or evaluation.
        '''

        self.image_paths = glob.glob(os.path.join(self.path, "*/*.jpg"))
        self.image_classes = []  # holds the class of every image (length = total num images)
        self.class_indices = {}  # mapping of class to every image in the class

        for image_path in self.image_paths:
            image_class = image_path.split(os.path.sep)[-2]
            self.image_classes.append(image_class)

            if image_class not in self.class_indices:
                self.class_indices[image_class] = []
            self.class_indices[image_class].append(self.image_paths.index(image_path))

        # we want to use an indices1 that balances malignant and benign (we know malignant is much less)
        # self.indices1 = np.arange(2 * len(self.class_indices[self.malignant]))

        # indices1 is the index for number of total images there are (0, 1, ..., n)
        self.indices1 = np.arange(len(self.image_paths))

        if self.shuffle_pairs:
            np.random.seed(int(time.time()))
            np.random.shuffle(self.indices1)
        else:
            # If shuffling is set to off, set the random seed to 1, to make it deterministic.
            np.random.seed(1)

        # make an array of random true/false values with length = the total number of images
        select_pos_pair = np.random.rand(len(self.image_paths)) < 0.5

        self.indices2 = []

        # iterate through every image index and its corresponding random true/false value
        for i, pos in zip(self.indices1, select_pos_pair):
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
            self.indices2.append(idx2)
        # vectorise so we can turn it to a tensor
        self.indices2 = np.array(self.indices2)

        # find the train/val/test split index
        self.train_val_index = int(len(self.image_paths) * self.train_portion)
        self.val_test_index = int(len(self.image_paths) * (self.train_portion + self.val_portion))

    def train_iter(self):
        return self.iterate(0, self.train_val_index)

    def val_iter(self):
        return self.iterate(self.train_val_index, self.val_test_index)

    def test_iter(self):
        return self.iterate(self.val_test_index, -1)

    def iterate(self, start_index, stop_index):
        # iterate will be a generator object that can be passed through the above methods
        for idx, idx2 in zip(self.indices1[start_index:stop_index], self.indices2[start_index:stop_index]):

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