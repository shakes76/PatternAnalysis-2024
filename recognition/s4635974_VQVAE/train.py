# In your train.py
from dataset import HipMRILoader
import modules
import torchvision.transforms as transforms

# Hyperparameters
num_epochs = 5
batch_size = 32
lr = 0.0002
num_hiddens = 128
num_residual_hiddens = 32
num_chanels = 1
num_embeddings = 512
dim_embedding = 64

# Directories for datasets
train_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'
test_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'
validate_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'

# Define your transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert numpy array to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    # transforms.RandomHorizontalFlip(),  # Random horizontal flip
    # transforms.RandomRotation(15),  # Random rotation within 15 degrees
])

# Get loaders
train_loader, validate_loader, train_variance = HipMRILoader(train_dir, validate_dir, test_dir,
                                                             batch_size=batch_size, transform=transform).get_loaders()

# Create model
model = modules.