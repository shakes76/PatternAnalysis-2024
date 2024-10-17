import torch
import torchvision
import torch.utils.data.dataloader as dataloader
import os
import matplotlib.pyplot as plt
import torchvision.transforms as tvtransforms
import torchvision.utils as vutils
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
                 batch_size = 32) -> None:
        self.device = device

        #paths to datasets
        adni_data_path_train = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'train'))
        adni_data_path_test = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'test'))

        transforms = tvtransforms.Compose([
            tvtransforms.Grayscale(), #grascale images 
            tvtransforms.Resize(64), #the next two lines decrease the resolution to 64x64
            tvtransforms.CenterCrop(64),
            tvtransforms.ToTensor(), #turn the datat into a tensor if its not already
            tvtransforms.Normalize(0.5,0.5)]) #normilze the data 0.5 beacuse values between 0-1 so 0.5 is just good general value

        #create datasets from image folder
        self.train_dataset = torchvision.datasets.ImageFolder(root=adni_data_path_train, transform=transforms)
        self.test_dataset = torchvision.datasets.ImageFolder(root=adni_data_path_test, transform=transforms)

        print(self.train_dataset)
        print(self.test_dataset)

        #create data loaders
        print("Creating DataLoaders...⏳") #setup data loadeers for training to make thins easier with batching
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
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
        plt.imshow(np.transpose(vutils.make_grid(images.to(self.device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
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