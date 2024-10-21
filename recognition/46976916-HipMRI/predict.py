import os
import torch


from dataset import load_data_2D




image_folder = 'keras_slices_train'
imageNames = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.nii.gz')]

images= load_data_2D(imageNames)

# Check the shape and content
print(f"Number of images loaded: {len(imageNames)}")
print(f"Shape of first image: {images[0].shape}")

#HyperParameters
LEARN_RATE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MEMORY = True
