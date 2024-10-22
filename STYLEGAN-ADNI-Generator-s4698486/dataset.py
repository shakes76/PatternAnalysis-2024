'''
The module reads and loads data.
The data is augmented and transformed during import for faster training.
'''

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from constants import image_height, image_width

'''
 Saves 5 images after the data transformation/augmentation and loading is complete and wrapped using dataloader.

 This is just a test to ensure that the images are being loaded as expected.
'''
def generate_sample_images(loader):    
    
    for i in range(5):
        features, _ = next(iter(loader))
        img = features[0].squeeze()
        plt.imshow(img, cmap="gray")
        save_image(img*0.5+0.5, f"aug_img_{i}.png") # Making sure to rescale image from [-1, 1] to [0, 1] for output.

'''
 Data Loader

 Data is the filepath for the data, provided in constants.py. This is the path to the folder containing both
 cognitive normal and alzheimer's disease brain scans for both the test and train scan.

 batchSize specifies the batches that we will 
'''
def get_data(data, batchSize):

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomVerticalFlip(p=0.5), # Augments data by randomly flipping horizontally 50% of the time - minimises overfitting.
            transforms.Grayscale(), # Converts images to grayscale if they are considered RGB.
            transforms.Resize((image_height, image_width), interpolation=transforms.InterpolationMode.BICUBIC), # BICUBIC is better than bilinear (usual interpolation method)
                                                                                                                # for preserving fine details in images, and minimising 
                                                                                                                # artifacts - at the cost of higher computation time. This is
                                                                                                                # a worthwhile trade, as image quality with bilinear isn't good
                                                                                                                # enough
            transforms.Normalize(mean=[0.5], std=[0.5])] # Normalises pixel data so that it is in the [0, 1] range instead of [0, 255]. Allows for tensor operations
                                                         # to run more smoothly.                                 
        )

    dataset = datasets.ImageFolder(root=data, transform=transform)

    loader = DataLoader(dataset, batchSize, shuffle=True) # shuffle to minimise overfitting to certain patterns in data order.

    generate_sample_images(loader) 
        
    return loader