"""
hyperparameters.py created by Matthew Lockett 46988133
"""
# The root directory for where the ADNI dataset is stored
# See torchvision ImageFolder class for the required dataset structure 
ROOT = r"C:\Users\Mathew\ADNI_Dataset"

# The relative path to a folder where all training output will be saved
SAVED_OUTPUT_DIR = r"recognition/StyleGAN-46988133/saved_output/"

# The desired output image resolution of the StyleGAN
DESIRED_IMAGE_SIZE = 256

# The number of channels of the images loaded into the model (1 = Greyscale)
NUM_CHANNELS = 1

# Represents the number of labels/classes used in the ADNI dataset (i.e. AD and CN)
LABEL_DIMENSIONS = 2

################################## Mapping Network ###############################

# The amount of fully connected layers within the StyleGAN Mapping Network
MAPPING_LAYERS = 8

# The size of the latent space vector and style vector within the StyleGAN Mapping Network
LATENT_SIZE = 512

# Controls the negative slope angle used for the leaky ReLu function
LRELU_SLOPE_ANGLE = 0.2

######################### Generator and Discriminator #############################

# Represents the size of the vector used to embed each class label
EMBED_DIMENSIONS = 8

# The amount of features used within the generator
GEN_FEATURE_SIZE = 512

# The amount of features used within the discriminator
DISC_FEATURE_SIZE = 64

################################## Progressive Growing ###########################

# The starting depth indicating the lowest resolution (8x8)
START_DEPTH = 0

# The max depth indicating the highest resolution (256x256)
MAX_DEPTH = 5

# Fade-In percentage used in blending old and new resolutions
FADE_IN_PERCENTAGE = 0.5

# The amount of steps per resolution
STEPS_PER_RESOLUTION = 5000

# The amount of epochs per image resolution
EPOCHS_PER_RESOLUTION = [40, 10, 10, 10, 10, 10]

# Different batch sizes required for different image resolutions
BATCH_SIZES = [256, 256, 128, 128, 32, 32, 32]

# The discriminator factors that influence the feature map sizes
GEN_FACTORS = [LATENT_SIZE, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64]

# The discriminator factors that influence the feature map sizes
DISC_FACTORS = [NUM_CHANNELS, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2]

################################## Training Loop #################################

# Mixing Regularisation ratio and probability
MIXING_RATIO = 0.5
MIXING_PROB = 0.5

# Set a seed for randomness for reproducibility
RANDOM_SEED = 999

# The learning rates used by Adam optimisers
GEN_LEARNING_RATE = 0.001
DISC_LEARNING_RATE = 0.001

################################## Helper Functions and Classes ##################

# Represents a small constant used to avoid division by zero
EPSILON = 1e-8