# Path
DATA = "/home/groups/comp3710/ADNI/AD_NC/"

# Hyper Parameters
epochs = 50            # Number of epochs to train
learning_rate = 0.001    # Learning rate
base_lr = 1e-5
channels = 1            # Number of channels (3 channels for the image if RGB)
batch_size = 32         # Batch Size
log_resolution = 8      # 256*256 image size as such 2^8 = 256 # use 2^7 for single gpu (faster)
image_height = 2**log_resolution    # The height of the generated image
image_width = 2**log_resolution     # The width of the generated image
z_dim = 256             # Size of the z latent space
w_dim = 256             # Size of the style vector latent space
lambda_gp = 10          # WGAN-GP set to standard value 10
                        

save = "save"           # Rename if changing a parameter and require a new dir for saved eg
