import random
import torch

# Hyperparameters

# UPDATE HERE to the file path to your data
data_root = "C:\Users\kylie\OneDrive\Documents\keras_png_slices_data\keras_png_slices_data\keras_png_slices_seg_train"

# Google colab residue
#data_root = "/content/drive/My Drive/COMP3710/assignment-two/keras_png_slices_data/keras_png_slices_data"

# Note that some of these hyperparams are redundant, but have not yet been phased out
workers = 2 # Number of workers for dataloader
nz = 100 # Size of z latent vector (i.e. size of generator input)
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator
beta1 = 0.5 # Beta1 hyperparameter for Adam optimizers
ngpu = 1 # Number of GPUs available. Use 0 for CPU mode.
epochs = 50 # Number of epochs
learning_rate = 0.001 # Learning rate
channels = 1 # 1 Channel for greyscale images, 3 for RGB.
batch_size = 32 # Number of images per training batch
image_size = 64 # Image size is 64 x 64 pixels
log_resolution = 7 # 256*256 image size, use 2^7 for single gpu (faster)
image_height = 2**log_resolution # height of the generated image
image_width = 2**log_resolution # width of the generated image
z_dim = 256 # Size of the z latent space [initialise to 256 for lower VRAM usage or faster training]
w_dim = 256 # Size of the style vector latent space [initialise to 256 for lower VRAM usage or faster training]
lambda_gp = 10 # WGAN-GP set to standard value 10
interpolation = "bilinear" # MRI scans are curvy, using bilinear may produce more edges at high resolution
save = "save" # Rename if changing a parameter and require a new dir for saved eg
load_model = False # Set to True if you want to load a pre-trained model

# CHANGE HERE the seed if wanted
# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)