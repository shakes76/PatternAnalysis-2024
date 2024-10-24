"""
dataset.py created by Matthew Lockett 46988133
"""
import os
import math
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import hyperparameters as hp


def load_ADNI_dataset(image_size, training_set):
    """
    Loads the ADNI dataset into the StyleGAN model, utilising a transform applied
    to every image to modify it to the desired resolution. The transform is specifically 
    catered for the greyscale images of the ADNI dataset.

    Param: image_size: The required image resolution used for training.
    Param: training_set: An indicator on whether the training set needs to be loaded. 
    Return: The ADNI dataset images split into batches and transformed into a form appropriate for
            the StyleGAN model.
    REF: This function was based on the PyTorch DCGAN tutoiral dataset loader example found at 
    REF: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html.
    REF: The progressive change in batch size and image resolution was inspired by the following website:
    REF: https://blog.paperspace.com/implementation-stylegan-from-scratch/
    """    
    # Define a transformation to be applied to the images upon being loaded in, and convert the images 
    # to the desired resolution.
    transform = transforms.Compose([
                                transforms.Grayscale(num_output_channels=hp.NUM_CHANNELS),
                                transforms.Resize((image_size, image_size)),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)), # Normalize grayscale images to between [-1, 1]
                            ])

    if training_set:
        # Load the ADNI dataset training images located at the root directory
        dataset = dset.ImageFolder(root=os.path.join(hp.ROOT, "Training Set"), transform=transform)
    else:
        # Load the ADNI dataset validation images located at the root directory
        dataset = dset.ImageFolder(root=os.path.join(hp.ROOT, "Validate Set"), transform=transform)

    # Calculate the required batch size based on the image resolution
    batch_size = hp.BATCH_SIZES[int(math.log2(image_size / 4))]

    # Split the images into batches for training
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader