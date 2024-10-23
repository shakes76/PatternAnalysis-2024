workers = 2 # Number of workers for dataloader
nz = 100 # Size of z latent vector (i.e. size of generator input)
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator
beta1 = 0.5 # Beta1 hyperparameter for Adam optimizers
ngpu = 1 # Number of GPUs available. Use 0 for CPU mode.
epochs = 300 # Number of epochs
learning_rate = 0.001 # Learning rate
channels = 1 # 1 Channel for greyscale images, 3 for RGB.
batch_size = 32 # Number of images per training batch
image_size = 64 # Image size is 64 x 64 pixels
log_resolution = 7 # Log of resolution
image_height = 2**log_resolution # asdf
image_width = 2**log_resolution # asdf
z_dim = 256 # asdf
w_dim = 256 # asdf
lambda_gp = 10 # asdf
interpolation = "bilinear" # asdf
save = "save" # asdf

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)