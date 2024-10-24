import torch
import os
import cv2
import random
import numpy as np
import os.path as osP
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import defaultdict
from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
# from torchdata.datapipes.iter import BucketBatcher, FileLister

## Using dataset path from rangpur, assuming this will be used for assessing as well
## Structured like below
## - DATASET_PATH/
    ## - test/
        ##  - AD/ (contains 4450 images)
        ##  - NC/ (contains 4540 images)
    ## - train/
        ##  - AD/ (contains 10400 images)
        ##  - NC/ (contains 11120 images)
DATASET_PATH_RANG = '/home/groups/comp3710/ADNI/AD_NC'
DATASET_PATH = '/Users/rorymacleod/Desktop/Uni/sem 2 24/COMP3710/Report/AD_NC'

## Start with generic batch size of 32, can change depending on model training procedure & results
BATCH_SIZE = 32
## Images naming convention is PatientID__MRISliceID.jpeg total of 20 images per patient
IMAGES_PER_PATIENT = 20
## Current image size is 256 x 240, rezise to 224 x 224 to better suit convolutions
IMAGE_SIZE = 224

## Define dataset transform
### send image to tensor
### basic resize & crop for convolutions
### basic normalisation for RGB inesity values per channel 
#### use 0.5 to place intensity values between [-1, 1]
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.RandomErasing(p=0.3, scale=(0.01, 10), ratio=(0.5, 2.0)),
    transforms.Normalize(mean=[0.25], std=[0.25]),
    transforms.ToTensor()
    ])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.Normalize(mean=[0.25], std=[0.25]),
    transforms.ToTensor()
    ])


class ADNI(Dataset):
    def __init__(self, path=DATASET_PATH, type="train", transform=None, val=False, ratio=0.8, tqdm=False):
        root = osP.join(path, type)
        self.path = root
        self.ad_path = osP.join(root, 'AD')
        self.nc_path = osP.join(root, 'NC')
        self.ad_processed_path = osP.join(root, 'AD_processed')
        self.nc_processed_path = osP.join(root, 'NC_processed')
        self.tdqm_disabled = tqdm

        self.preprocess()

        self.ad_images =  [osP.join(self.ad_processed_path, f) for f in os.listdir(self.ad_processed_path)]
        self.nc_images = [osP.join(self.nc_processed_path, f) for f in os.listdir(self.nc_processed_path)]

        self.images = self.ad_images + self.nc_images
        self.labels = [1] * len(self.ad_images) + [0] * len(self.nc_images)

        self.val = val
        self.ratio = ratio
        if transform == "train":
            self.transform = TRAIN_TRANSFORM
        else:
            self.transform = TEST_TRANSFORM

        self.mask = self.make_mask()
        
    def make_mask(self):
        random.seed(69)
        if self.ratio == 1:
            return [True] * len(self.images)
        mask = [random.random() < self.ratio for _ in range(len(self.images))]
        return mask if not self.val else [not m for m in mask]

    def preprocess(self):
        if not osP.exists(self.ad_processed_path):
            print("Processing AD")
            os.makedirs(self.ad_processed_path)
            self.process_dir(self.ad_path, self.ad_processed_path)

        if not osP.exists(self.nc_processed_path):
            print("Processing NC")
            os.makedirs(self.nc_processed_path)
            self.process_dir(self.nc_path, self.nc_processed_path)

    def process_dir(self, in_path, out_path):
        for filename in tqdm(os.listdir(in_path), disable=self.tdqm_disabled):
            if filename.lower().endswith('.jpeg'):
                input_path = osP.join(in_path, filename)
                output_path = osP.join(out_path, filename)
                self.process_image(input_path, output_path)

    def process_image(self, in_path, out_path):
        img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        crop = self.crop_brain(img)

        h, w = crop.shape

        scale = 210 / max(h, w)

        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        top = (210-new_h) // 2
        bottom = (210 - new_h - top)
        left = (210 - new_w) // 2
        right = (210 - new_w - left)

        pad = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        cv2.imwrite(out_path, pad)


    def crop_brain(self, image):
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(binary)

        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y+h, x:x+w]
    
    def __len__(self):
        return sum(self.mask)
    
    def __getitem__(self, id):
        true_id = [i for i, m in enumerate(self.mask) if m][id]
        img_path = self.images[id]
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)


class ADNITest(Dataset):
    def __init__(self, path=DATASET_PATH, tqdm=False):
        self.path = osP.join(path, 'test')
        self.ad_path = osP.join(self.path, 'AD')
        self.nc_path = osP.join(self.path, 'NC')
        self.ad_processed_path = osP.join(self.path, 'AD_processed')
        self.nc_processed_path = osP.join(self.path, 'NC_processed')
        self.tqdm_disabled = tqdm
        self.transform = TEST_TRANSFORM

        self.preprocess()

        self.ad_groups = self.group_images(self.ad_processed_path, label=1)
        self.nc_groups = self.group_images(self.nc_processed_path, label=0)

        self.groups = self.ad_groups + self.nc_groups 

    def preprocess(self):
        if not osP.exists(self.ad_processed_path):
            print("Processing AD")
            os.makedirs(self.ad_processed_path)
            self.process_dir(self.ad_path, self.ad_processed_path)

        if not osP.exists(self.nc_processed_path):
            print("Processing NC")
            os.makedirs(self.nc_processed_path)
            self.process_dir(self.nc_path, self.nc_processed_path)

    def process_dir(self, in_path, out_path):
         for filename in tqdm(os.listdir(in_path), disable=self.tdqm_disabled):
            if filename.lower().endswith('.jpeg'):
                input_path = osP.join(in_path, filename)
                output_path = osP.join(out_path, filename)
                self.process_image(input_path, output_path)

    def process_image(self, in_path, out_path):
        img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        crop = self.crop_brain(img)

        h, w = crop.shape

        scale = 210 / max(h, w)

        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        top = (210-new_h) // 2
        bottom = (210 - new_h - top)
        left = (210 - new_w) // 2
        right = (210 - new_w - left)

        pad = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        cv2.imwrite(out_path, pad)

    def crop_brain(self, image):
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(binary)

        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y+h, x:x+w]
    
    def group_images(self, processed_path, label):
        groups = []
        group_dict = defaultdict(list)
        
        for filename in os.listdir(processed_path):
            if filename.lower().endswith('.jpeg'):
                group_number = filename.split('_')[0]
                group_dict[group_number].append(filename)
        
        for group_number, filenames in group_dict.items():
            filenames = sorted(filenames, key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)

            groups.append({
                "group_number": group_number,
                "filenames": filenames,
                "label": label
            })
        
        return groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, id):
        info = self.groups[id]
        number = info['group_number']
        filenames = info['filenames']
        label = info['label']

        stack = []

        processed_path = self.ad_processed_path if label == 1 else self.nc_processed_path
        
        for filename in filenames:
            img_path = os.path.join(processed_path, filename)
            img = Image.open(img_path).convert('L')
            if self.transform:
                img = self.transform(img)

            stack.append(img)
        
        stack = np.stack(stack, axis=0)
        return torch.tensor(stack, dtype=torch.float32), torch.tensor(label).float()



# """
# Calculate mean and standard deviation for transform
# """
# def calc_mean_std(dataloader):
#     mean = 0.0
#     mean_sqr = 0.0
#     x = 0
#     for data, _ in dataloader:
#         x += data.size(0)
#         mean += data.sum(dim=(0,2,3))
#         mean_sqr += (data ** 2).sum(dim=(0,2,3))

#     mean /= x * data.size(2) * data.size(3)
#     mean_sqr /= x * data.size(2) * data.size(3)
#     std = (mean_sqr - mean ** 2).sqrt()
#     return mean, std

# def get_ids(path):
#     files = [osP.basename(file) for _, _, filenames in os.walk(path) for file in filenames]
#     ids = list(set([file.split('_')[0] for file in files]))
#     return ids

# def create_dataloaders(batch_size=BATCH_SIZE, path=DATASET_PATH):
#     sampler_dataset = datasets.ImageFolder(root=path+"/train", transform=RESIZE_TRANSFORM)
#     sampler = SubsetRandomSampler(torch.randperm(len(sampler_dataset))[:1000])
#     sample_loader = DataLoader(sampler_dataset, batch_size=batch_size, sampler=sampler)

#     mean, std = calc_mean_std(sample_loader)

#     train_transform = transforms.Compose([
#         transforms.Grayscale(),
#         transforms.Resize(IMAGE_SIZE),
#         transforms.CenterCrop(IMAGE_SIZE),
#         transforms.RandomVerticalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std),
#     ])

#     test_transform = transforms.Compose([
#         transforms.Grayscale(),
#         transforms.Resize(IMAGE_SIZE),
#         transforms.CenterCrop(IMAGE_SIZE),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std)
#     ])

#     train = datasets.ImageFolder(root=path+"/train", transform=train_transform)
#     test = datasets.ImageFolder(root=path+"/test", transform=test_transform)

#     ids = get_ids(path+"/train")
#     train_indices = [i for i, (_path, _) in enumerate(train.samples) if _path.split('\\')[-1].split('_')[0] in ids]
#     train_subset = Subset(train, train_indices)

#     train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
#     test_dataload = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)
#     return train_dataloader, test_dataload



# def load_data(path=DATASET_PATH, transform=DATASET_TRANSFORM, batch_size=BATCH_SIZE,
#                imgs_per_patient=IMAGES_PER_PATIENT, testing=False):
#     if testing:
#         test_images = datasets.ImageFolder(root=osP.join(path, "test"), transform=DATASET_TRANSFORM)
#         return test_images, len(list(test_images)), None

#     # create training datasets including lables with their respective class
#     AD_files = FileLister(root=osP.join(path, "AD", "train"), 
#                           masks="*.jpeg", recusive=False).map(label_file)
#     NC_files = FileLister(root=osP.join(path, "NC", "train"), 
#                           masks="*.jpeg", recusive=False).map(label_file)

#     # batch data, grouped by patient ID 
#     # buffer shuffle used to shuffle batches corresponding to patient within entire bucket
#     AD_batch = AD_files.bucketbatch(use_in_batch_shuffle=False, 
#                                     batch_size=imgs_per_patient, sort_key=sort_patients)
#     NC_batch = NC_files.bucketbatch(use_in_batch_shuffle=False, 
#                                     batch_size=imgs_per_patient, sort_key=sort_patients)
    
#     # stratified split of AD & NC images to get validation data
#     val_size = 0.2
#     AD_train, AD_val = AD_batch.random_split(weights={
#                                                 "train": 0.8,
#                                                 "validation": 0.2},
#                                                 total_length=len(list(AD_batch)),
#                                                 seed=1)
#     NC_train, NC_val = NC_batch.random_split(weights={
#                                                 "train": 0.8,
#                                                 "validation": 0.2},
#                                                 total_length=len(list(NC_batch)),
#                                                 seed=2)

#     # combine AD & NC class splits
#     # unbatch data 
#     # shuffle data
#     train_data = AD_train.concat(NC_train).unbatch().shuffle()
#     val_data = AD_val.concat(NC_val).unbatch().shuffle()
#     num_train_datapoints = len(list(train_data))

#     # apply sharing filter to data 
#     # open images and apply transforms to images
#     train_images = train_data.sharding_filter().map(open_image).map(apply_tf)
#     val_images = val_data.sharding_filter().map(open_image).map(apply_tf)

#     return train_images, num_train_datapoints, val_images

# """
# Method to apply transform to single image
# """
# def apply_tf(image_data, tf=DATASET_TRANSFORM):
#     image, class_name = image_data
#     return tf(image), class_name

# def open_image(data):
#     filename, class_name = data
#     return Image.open(filename).convert("RGB"), class_name

# """
# Method to determine the class label for the given file based on its filename.

# Params:
#     filename (str): file name for given input image
# Returns:
#     filename (str): file name for given input image
#     class_name (int): the class for the image, 0 = AD, 1 = NC
# Throws an exception if the class label cannot be determined from provided filename
# """
# def label_file(filename):
#     split = filename.split("AD_NC")
#     if split[-1].find("AD") != -1:
#         class_name = 0
#     elif split[-1].find("NC") != -1:
#         class_name = 1
#     else:
#         return Exception()
#     return filename, class_name

# """
# Sorts the provided selection (bucket) of images based on filenames, such that 
# images of the same patient are grouped in the same batch

# Assumes all image filenames within a bucket are from the same directory, to ensure
# correct lexicographical sorting order. Grouping is on patient ID.

# Params:
#     bucket (torch object): a collection (bucket) of images, including filenames
# Returns:
#     bucket (torch object): the bucket after having been sorted by filename, in 
#     lexicographical order   
# """
# def sort_patients(bucket):
#     return sorted(bucket)
