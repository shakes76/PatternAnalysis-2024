import nibabel as nib
import numpy as np
import glob
import torchio as tio
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random



class load_data_3D(DataLoader) :
    '''
    Load medical image data from names.

    '''
    def __init__(self, image_path, label_path):
        self.images = []
        self.labels = []
        #retrieve path from dataset
        for f in sorted(glob.iglob(image_path)): 
            self.images.append(f)
        for f in sorted(glob.iglob(label_path)):
            self.labels.append(f)
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)
    
    #open files
    def __getitem__(self, idx): 
        image_p = self.images[idx]
        label_p = self.labels[idx]

        image = nib.load(image_p)
        image = np.asarray(image.dataobj)

        label = nib.load(label_p)
        label = np.asarray(label.dataobj)
        
        image = self.totensor(image)
        image = image.unsqueeze(0)
        image = image.data

        label = self.totensor(label)
        label = label.unsqueeze(0)
        label = label.data
        
        return image, label
    

class augmentation:
    def __init__(self) :

        self.crop = tio.CropOrPad((16,32,32))

        #Flip/augment the data 
        self.rand_flip_0 = tio.transforms.RandomFlip(0, flip_probability = 1) 
        self.rand_flip_1 = tio.transforms.RandomFlip(1, flip_probability = 1)
        self.rand_flip_2 = tio.transforms.RandomFlip(2, flip_probability = 1)

        #Additional augmentation methods
        bias_field = tio.transforms.RandomBiasField()
        blur = tio.transforms.RandomBlur()
        spike = tio.transforms.RandomSpike()

        prob = {}
        prob[bias_field] = 0.4
        prob[blur] = 0.3
        prob[spike] = 0.3

        #Randomly choose an augmentation method  
        self.oneof = tio.transforms.OneOf(prob) 

    def augment(self, image, label):

        generate = random.randint(0,3)

        if generate == 0:
            image = self.rand_flip_0(image)
            label = self.rand_flip_0(label)
        elif generate == 1:
            image = self.rand_flip_1(image)
            label = self.rand_flip_1(label)
        elif generate == 2:
            image = self.rand_flip_2(image)
            label = self.rand_flip_2(label)
        elif generate == 3:
            image = self.crop(image)
            label = self.crop(label)

        image = self.oneof(image)
        
        return image, label
