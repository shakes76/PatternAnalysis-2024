'''
[desc]
contains a class to represent the dataset
to be used to get all the dataloaders
and datasets for the data 

also handles loading in the data
and train, test and validation splits

@author Jamie Westerhout
@project Stable Diffusion
@date 2024
'''
import torch
import torchvision
import torch.utils.data.dataloader as dataloader
import os
import matplotlib.pyplot as plt
import torchvision.transforms as tvtransforms
import torchvision.utils as tvutils
import torch.utils as utils
import numpy as np

#=======================================================
# key variables
#=======================================================
IMAGES = 0
LABELS = 1
#=======================================================

class Dataset():
    '''
        class for loading data from the ADNI dataset
    '''
    def __init__(self, 
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 batch_size = 128) -> None:
        self.device = device
        self.image_size = 200

        val_set_created = False
        #create folder for validation set
        if not os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'val'))):
            os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'val')))
        else:
            val_set_created = True
        if not os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'val', 'AD'))):
            os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'val', 'AD')))
        else:
            val_set_created = True
        if not os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'val', 'NC'))):
            os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'val', 'NC')))
        else:
            val_set_created = True

        #paths to datasets
        adni_data_path_train = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'train'))
        adni_data_path_test = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'test'))
        adni_data_path_val = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'val'))

        transforms = tvtransforms.Compose([
            tvtransforms.Grayscale(), #grascale images 
            tvtransforms.Resize(self.image_size), #the next two lines decrease the resolution to 64x64
            tvtransforms.CenterCrop(self.image_size),
            tvtransforms.ToTensor(), #turn the datat into a tensor if its not already
            tvtransforms.Normalize(0.5,0.5)]) #normilze the data 0.5 beacuse values between 0-1 so 0.5 is just good general value

        #create datasets from image folder
        self.train_dataset = torchvision.datasets.ImageFolder(root=adni_data_path_train, transform=transforms)
        self.test_dataset = torchvision.datasets.ImageFolder(root=adni_data_path_test, transform=transforms)

        #move roughly 10% of the test images to val to be a validation set
        test_len = len(self.test_dataset)
        val_len = test_len // 100
        patients_seen = set()
        if not val_set_created:
            for i in os.listdir(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'test', 'AD'))):
                patient = i.split("_")[0]
                if len(patients_seen) <= val_len:
                    patients_seen.add(patient)
                if patient in patients_seen:
                    os.rename(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'test', 'AD', i)),
                            os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'val', 'AD', i)))
            for i in os.listdir(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'test', 'NC'))):
                patient = i.split("_")[0]
                if len(patients_seen) <= val_len:
                    patients_seen.add(patient)
                if patient in patients_seen:
                    os.rename(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'test', 'NC', i)),
                            os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'val', 'NC', i)))

        self.val_dataset = torchvision.datasets.ImageFolder(root=adni_data_path_val, transform=transforms)

        print(self.train_dataset)
        print(self.test_dataset)
        print(self.val_dataset)

        #create data loaders
        print("Creating DataLoaders...⏳") #setup data loadeers for training to make thins easier with batching
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True)
        print("DataLoaders Created ✅")

    def sample(self):
        '''
            display a sample from the dataset
            using matplotlib imshow
        '''
        #get sample of data
        images = next(iter(self.train_dataloader))
        print(len(images[IMAGES]), len(images[LABELS]))
        images = images[IMAGES]

        #visualise the sample
        print("Showing Data Sample...⏳")
        print(images.shape)
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("64 Samples of Training Images")
        plt.imshow(np.transpose(tvutils.make_grid(images.to(self.device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
        print("Data Sample shown ✅")
    
    def get_train(self):
        '''
            return the dataloader for the training data
        '''
        return self.train_dataloader
    
    def get_test(self):
        '''
            return the test dataloader for the testing data
        '''
        return self.test_dataloader
    
    def get_val(self):
        '''
            return the validation dataloader for the testing data
        '''
        return self.val_dataloader
    
    def get_train_raw(self):
        '''
            return the raw training data
        '''
        return self.train_dataset
    
    def get_test_raw(self):
        '''
            return the raw testing data
        '''
        return self.test_dataset

    def get_val_raw(self):
        '''
            return the raw validation data
        '''
        return self.val_dataset