''' Wil be used to store all the configuration variables for the project '''


TRAIN = True # When Training
LOAD = False # Load Checkpoint from LOAD_CHECK
SAVE_MODEL = False # Saves model to save_path
TEST = False # TEST loaded model usually when using checkpoint model


# Data path
AD_train = "/home/groups/comp3710/ADNI/AD_NC/train/AD"
AD_test = "/home/groups/comp3710/ADNI/AD_NC/test/AD"
NC_train = "/home/groups/comp3710/ADNI/AD_NC/train/NC"
NC_test = "/home/groups/comp3710/ADNI/AD_NC/test/NC"

BRAIN = 'AD' # Change for different type of Brain

# ----Image Parameters----
#
log_resolution = 7 # 256x256 image size as such 2^8 = 256 # use 2^7 for single gpu (faster)
IMAGE_SIZE = (256, 240) 
image_height = 2**log_resolution    # The height of the generated image
image_width = 2**log_resolution     # The width of the generated image
#------------------------
BATCH_SIZE = 16
LEARNING_RATE = 0.001
CHANNELS = 1  
interpolation = 'bilinear'

N_FEATURES = 256
MAX_FEATURES = 512
EPOCHS = 41 # ! Training Epoch

#------ New code ------
lambda_gp = 10 # gradient penalty
W_DIM = 256
Z_DIM = 256 #


# ----------------------

save_path = '/home/Student/s4653241/checkpoints/'
LOAD_CHECK= "/home/Student/s4653241/checkpoints/checkpoint_StyleGAN2_Epoch26.pth" # LOAD checkpoint