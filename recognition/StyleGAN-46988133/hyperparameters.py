"""
hyperparameters.py created by Matthew Lockett 46988133
"""
# The root directory for where the ADNI dataset is stored
# See torchvision ImageFolder class for the required dataset structure 
ROOT = r"C:\Users\Mathew\AD_NC"

# The relative path to a folder where all figures will be saved
SAVED_FIGURES_DIR = r"recognition/StyleGAN-46988133/saved_figures/"

# The IMAGE_SIZExIMAGE_SIZE pixel dimension of the images loaded into the model
IMAGE_SIZE = 256

# The number of channels of the images loaded into the model (1 = Greyscale)
NUM_CHANNELS = 1

# The total number of images trained on the model at any given time
BATCH_SIZE = 128

################################## Mapping Network ###############################

# The amount of fully connected layers within the StyleGAN Mapping Network
MAPPING_LAYERS = 8

# The size of the latent space vectors within the StyleGAN Mapping Network
LATENT_SIZE = 512

# Controls the negative slope angle used for the leaky ReLu function
LRELU_SLOPE_ANGLE = 0.2