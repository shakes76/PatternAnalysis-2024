import os
import torch

import tqdm

from modules import UNET
from dataset import ProstateCancerDataset
import matplotlib.pyplot as plt


#image_folder = 'keras_slices_train'
#imageNames = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.nii.gz')]

#images= load_data_2D(imageNames)

# Check the shape and content
#print(f"Number of images loaded: {len(imageNames)}")
#print(f"Shape of first image: {images[0].shape}")

#HyperParameters
LEARN_RATE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MEMORY = True


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    

def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    image_dir = 'C:/Users/baile/OneDrive/Desktop/HipMRI_study_keras_slices_data/keras_slices_train'
    seg_dir = 'C:/Users/baile/OneDrive/Desktop/HipMRI_study_keras_slices_data/keras_slices_seg_train'

    dataset = ProstateCancerDataset(image_dir, seg_dir)

    image, ground = dataset[0]

    print("Image shape:", image.shape)  # Should print something like (1, H, W) where H, W are image dimensions
    print("Ground truth shape:", ground.shape)
    plt.imshow(image, cmap='gray')  # Use cmap='gray' for grayscale display
    plt.title(f'Image 0')
    plt.axis('off')  # Turn off axis labels
    plt.show()

if __name__ == "__main__":
    main()