import nibabel as nib
import numpy as np
import glob
import torch
import torchio as tio
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random



class load_data_3D(Dataset) :
    '''
    Load medical image data from specified paths.
    This class inherits from DataLoader and is designed to handle 3D medical images
    and their corresponding labels for tasks such as segmentation.
    '''
    def __init__(self, image_path, label_path, augment=False):

        # Initialize empty lists to store image and label file paths
        self.images = []
        self.labels = []
        
        # Retrieve image file paths from the specified image directory using glob
        for f in sorted(glob.iglob(image_path)): 
            self.images.append(f)
        
        # Retrieve label file paths from the specified label directory using glob
        for f in sorted(glob.iglob(label_path)):
            self.labels.append(f)
        
        # Initialize a transformation to convert images to tensors
        self.totensor = transforms.ToTensor()

        # Set augmentation flag and define augmentations
        self.augment = augment

        if self.augment:
            # Define 3D augmentations 
            self.augmentations = tio.Compose([
                tio.RandomFlip(axes=(0, 1, 2)),        # Randomly flip along any axis
                tio.RandomAffine(scales=(0.9, 1.1),    # Random scaling
                                 degrees=15),          # Random rotation
                tio.RandomGamma(log_gamma=(-0.3, 0.3)),# Random intensity adjustment
                tio.OneOf({                            # Randomly apply one of the following:
                    tio.RandomElasticDeformation(): 0.5,  # Elastic deformation
                    tio.RandomAnisotropy(): 0.5            # Simulate anisotropic scanning resolution
                })
            ])


    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.images)
    
    
    def __getitem__(self, idx): 
        # Retrieve the image and label file paths based on the index
        image_p = self.images[idx]
        label_p = self.labels[idx]

        # Load the image/label data using nibabel and convert to a numpy array
        image = nib.load(image_p)
        image = np.asarray(image.dataobj)
        label = nib.load(label_p)
        label = np.asarray(label.dataobj)

        # Optionally apply augmentations
        if self.augment:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=torch.tensor(image, dtype=torch.float32).unsqueeze(0)),
                label=tio.LabelMap(tensor=torch.tensor(label, dtype=torch.float32).unsqueeze(0))
            )
            augmented = self.augmentations(subject)
            image = augmented['image'].numpy()[0]  # Convert back to numpy (3D image)
            label = augmented['label'].numpy()[0]  # Convert back to numpy (3D label)

        
        # Convert the image/label numpy array to a tensor
        image = self.totensor(image)
        label = self.totensor(label)

        # Add a channel dimension to the image/label tensor (needed for 3D input)
        image = image.unsqueeze(0)
        image = image.data
        label = label.unsqueeze(0)
        label = label.data
        
        # Return the image and label tensors
        return image, label
    

class augmentation:
    """
    A class for applying various augmentations to medical image data.

    This class includes methods for random flipping, cropping, and adding noise to images and labels
    to enhance the diversity of the training dataset.

    """
    def __init__(self) :

        #Flip/augment the data 
        self.rand_flip_0 = tio.transforms.RandomFlip(0, flip_probability = 1) 
        self.rand_flip_1 = tio.transforms.RandomFlip(1, flip_probability = 1)
        self.rand_flip_2 = tio.transforms.RandomFlip(2, flip_probability = 1)
        self.crop = tio.CropOrPad((16,32,32))

        #Additional augmentation methods
        flip = tio.transforms.RandomFlip(axes=(0,))
        zoom = tio.transforms.RandomNoise(mean=0.0, std=0.1)
        nothing = tio.transforms.RandomFlip(2, flip_probability = 0)
        bias_field = tio.transforms.RandomBiasField()
        blur = tio.transforms.RandomBlur()
        spike = tio.transforms.RandomSpike()

        prob = {}
        prob[nothing] = 0.1
        prob[bias_field] = 0.3
        prob[blur] = 0.3
        prob[spike] = 0.3

        #Randomly choose an augmentation method  
        self.oneof = tio.transforms.OneOf(prob) 

    def augment(self, image, label):

        # Randomly generate an integer to determine which augmentation to apply
        generate = random.randint(0,2)

        # Apply the selected augmentation based on the generated integer
        if generate == 0:
            image = self.rand_flip_0(image)
            label = self.rand_flip_0(label)
        elif generate == 1:
            image = self.rand_flip_1(image)
            label = self.rand_flip_1(label)
        elif generate == 2:
            image = self.rand_flip_2(image)
            label = self.rand_flip_2(label)

        image = self.oneof(image)
        
        return image, label
