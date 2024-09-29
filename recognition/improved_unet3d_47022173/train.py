from dataset import *
import torch
from torch.utils.data import DataLoader, random_split
import torchio as tio
from utils import *

images_path = "./data/semantic_MRs_anon/"
masks_path = './data/semantic_labels_anon/'
batch_size = 1
shuffle = True
num_workers = 1
in_channels = 1 # greyscale
n_classes = 5 #5 different values in mask
base_n_filter = 8

if __name__ == '__main__':
    # Load and process data
    transforms = tio.Compose([
        tio.RescaleIntensity((0, 1)),
        tio.RandomFlip(),
        tio.RandomAffine(degrees=10),
        tio.RandomElasticDeformation(),
        tio.CropOrPad((256, 256, 128)),
        tio.ZNormalization(),
    ])

    dataset = ProstateDataset3D(images_path, masks_path, transforms)
    fixed_gen = torch.Generator().manual_seed(SEED)
    train_dataset, test_dataset = random_split(dataset, [TRAIN_SIZE, 1 - TRAIN_SIZE], generator=fixed_gen)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    # Model
    

