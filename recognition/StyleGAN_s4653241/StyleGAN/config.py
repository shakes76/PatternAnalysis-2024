''' Wil be used to store all the configuration variables for the project '''






# Data path
# AD_train = "/home/groups/comp3710/ADNI/AD_NC/train/AD"
# AD_test = "/home/groups/comp3710/ADNI/AD_NC/test/AD"
# NC_train = "/home/groups/comp3710/ADNI/AD_NC/train/NC"
# NC_test = "/home/groups/comp3710/ADNI/AD_NC/test/NC"

AD_train = "/Users/solomontjipto/Documents/COMP3710/PatternAnalysis-2024/recognition/Data/AD_NC/train/AD"
AD_test = "/Users/solomontjipto/Documents/COMP3710/PatternAnalysis-2024/recognition/Data/AD_NC/test/AD"
NC_train = "/Users/solomontjipto/Documents/COMP3710/PatternAnalysis-2024/recognition/Data/AD_NC/train/NC"
NC_test = "/Users/solomontjipto/Documents/COMP3710/PatternAnalysis-2024/recognition/Data/AD_NC/test/NC"

BRAIN = 'NC'

# ----Image Parameters----
#
log_resolution = 7 # 256x256 image size as such 2^8 = 256 # use 2^7 for single gpu (faster)
IMAGE_SIZE = (256, 240) # this resizes the image
image_height = 2**log_resolution    # The height of the generated image
image_width = 2**log_resolution     # The width of the generated image
#------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
CHANNELS = 1  # Single channel image !!!!
interpolation = 'bilinear'
channels = 1

N_FEATURES = 256
MAX_FEATURES = 512
EPOCHS = 10

#------ New code ------
lambda_gp = 10
W_DIM = 256
Z_DIM = 256 # Should I change this to 256?
w_dim = 256

# ----------------------

save_path = 'recognition/checkpoints/'
load_path = ''
