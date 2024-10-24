"""
dataset.py created by Matthew Lockett 46988133
"""
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import hyperparameters as hp

# Define a transformation to be applied to the images upon being loaded in
# REF: This transform was inspired by the PyTorch DCGAN tutoiral found at:
# REF: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html.
transform = transforms.Compose([
                                transforms.Grayscale(num_output_channels=hp.NUM_CHANNELS),
                                transforms.Resize((hp.IMAGE_SIZE, hp.IMAGE_SIZE)),
                                transforms.CenterCrop(hp.IMAGE_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)), # Normalize grayscale images to between [-1, 1]
                            ])

def load_ADNI_dataset(training_set):
    """
    Loads the ADNI dataset into the StyleGAN model, utilising a predefined transform applied
    to every image. The transform is specifically catered for the greyscale images of the ADNI 
    dataset.

    Param: training_set: An indicator on whether the training set needs to be loaded. 
    Return: The ADNI dataset images split into batches and transformed into a form appropriate for
            the StyleGAN model.
    REF: This function was based on the PyTorch DCGAN tutoiral dataset loader example found at 
    REF: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html.
    """    
    if training_set:
        # Load the ADNI dataset training images located at the root directory
        dataset = dset.ImageFolder(root=os.path.join(hp.ROOT, "Training Set"), transform=transform)
    else:
        # Load the ADNI dataset validation images located at the root directory
        dataset = dset.ImageFolder(root=os.path.join(hp.ROOT, "Validate Set"), transform=transform)

    # Split the images into batches for training
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=hp.BATCH_SIZE, shuffle=True)

    return train_loader