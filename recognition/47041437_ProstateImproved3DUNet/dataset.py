import nibabel as nib
import numpy as np
import glob
import torchio as tio
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


'''
Class for loading data from nii.gz files and prepocessing the data
'''
class load_data_3D(DataLoader) :
    def __init__(self, image_path, label_path):
        self.inputs = []
        self.labels = []
        #retrieve path from dataset
        for f in sorted(glob.iglob(image_path)): 
            self.inputs.append(f)
        for f in sorted(glob.iglob(label_path)):
            self.labels.append(f)
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.inputs)
    
    #open files
    def __getitem__(self, idx): 
        image_p = self.inputs[idx]
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
    

'''Class for data augmentation'''
class augmentation:
    def __init__(self) :
        self.shrink = tio.CropOrPad((16,32,32))
        self.flip0 = tio.transforms.RandomFlip(0, flip_probability = 1) #flip the data randomly
        self.flip1 = tio.transforms.RandomFlip(1, flip_probability = 1)
        self.flip2 = tio.transforms.RandomFlip(2, flip_probability = 1)

        nothing = tio.transforms.RandomFlip(0, flip_probability = 0)
        bias_field = tio.transforms.RandomBiasField()
        blur = tio.transforms.RandomBlur()
        spike = tio.transforms.RandomSpike()
        prob = {}
        prob[nothing] = 0.7
        prob[bias_field] = 0.1
        prob[blur] = 0.1
        prob[spike] = 0.1
        self.oneof = tio.transforms.OneOf(prob) #randomly choose one augment method from the three 

    def crop_and_augment(self, image, label):
        image = self.shrink(image)
        label = self.shrink(label)
        image = self.oneof(image)
        
        return image, label
