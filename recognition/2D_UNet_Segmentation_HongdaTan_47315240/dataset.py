import os
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomResizedCrop, Resize, Normalize, CenterCrop
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


class CIFAR10DataModule(LightningDataModule):
    """Pytorch_lighning module that will handle loading data into each epoch
        Will also handle downloading and preproccessing of data"""
    def __init__(
        self, 
        batch_size,
        image_size,
        in_channels,
        num_workers=1):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = os.environ.get("PATH_DATASETS", "Data/")

        #Check root is correct
        print(self.data_root)

        #Setup transfroms
        #Transfrom for both test and validation sets
        self.test_transform = Compose([ToTensor(), cifar10_normalization()])

        #transform for traing set
        self.train_transform = Compose([ 
                                RandomHorizontalFlip(),
                                RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.0)),
                                ToTensor(),
                                cifar10_normalization()])


        #Splitting trickery as val set shouldnt use train_transfrom
        #but need to keep test and val sperate
        pl.seed_everything(42)
        train_set = CIFAR10(root=self.data_root, train=True, transform=self.train_transform, download=True)
        self.train, _ = random_split(train_set, [45000, 5000])

        val_set = CIFAR10(root=self.data_root, train=True, transform=self.test_transform, download=True)
        _, self.test = random_split(val_set, [45000, 5000])

        self.test = CIFAR10(root=self.data_root, train=False, transform=self.test_transform, download=True)

    #Pytorch_lightning functions for loading split data
    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, num_workers=self.num_workers, shuffle=True)    

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size, num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, shuffle=False)