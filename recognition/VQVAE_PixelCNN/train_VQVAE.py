"""
This module is use train the models.
"""

# Importing libraries and modules
import os
import torch
from torch.optim.lr_scheduler import MultiStepLR
from skimage.metrics import structural_similarity as compute_ssim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import *
from modules import *

# Hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 128
NUM_RESIDUAL_LAYER = 2
RESIDUAL_HIDDEN_DIM = 32
NUM_EMBEDDINGS = 128
EMBEDDING_DIM = 128
COMMITMENT_COST = 0.25
LEARNING_RATE = 1e-2
NUM_EPOCH_VQVAE = 1500

def train():

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create image saving path
    if not os.path.exists("./Output"):
        os.mkdir("./Output")

    if not os.path.exists("./Model"):
        os.mkdir("./Model")

    tqdm_bar = tqdm(range(NUM_EPOCH_VQVAE))

    # Initialize Parameters
    i = 0
    ssim = None
    previous_loss = 1
    recon_img = None

    # Load Model, Loss function and multistep scheduler
    model = Model(HIDDEN_DIM, RESIDUAL_HIDDEN_DIM, NUM_RESIDUAL_LAYER, NUM_EMBEDDINGS, EMBEDDING_DIM,
                  COMMITMENT_COST).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    scheduler = MultiStepLR(optimizer, milestones=[1000, 9000], gamma=0.1)

    model.train()

    # Define lists for plotting
    training_times_under_3000 = []
    loss_under_3000_train = []
    training_times_beyond_3000 = []
    loss_beyond_3000_train = []

    # Use tqdm bar as for loop starter
    for eq in tqdm_bar:
        train_img = next(iter(dataloader))

        # Fit data and model into device
        train_img = train_img.to(device)
        model = model.to(device)

        # Fit data into model
        vq_loss, recon, perplexity, _ = model(train_img)
        loss = F.mse_loss(recon, train_img) + vq_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        i += 1

        if i <= 3000:
            # Record the loss and training epochs
            training_times_under_3000.append(i)
            loss_under_3000_train.append(loss.cpu().detach().item())
        else:
            training_times_beyond_3000.append(i)
            loss_beyond_3000_train.append(loss.cpu().detach().item())

        # Only record the smaller loss
        if loss < previous_loss:
            recon_img = recon
            trained_image = recon_img.cpu().detach().numpy()
            original_image = train_img.cpu().detach().numpy()
            ssim = compute_ssim(trained_image[0][0], original_image[0][0], data_range=2.0)

            previous_loss = loss
            print('Loss: {}'.format(loss))

            # Save generated images under the folder, all the Images have loss and ssim as their name
            loss_img = loss.cpu().detach().item()
            save_image(recon, "./Output/No_{}_img_Loss_{}_SSIM_{}%.jpg".
                       format(i, loss_img, ssim * 100))
            # Save the generated model under the folder
            torch.save(model.state_dict(), "./Model/Vqvae.pth")

        # Show the running status every 10 epochs(show it is still working and visualisation of loss changing)
        if i % 10 == 0:
            tqdm_bar.set_description('loss: {}'.format(loss))

    # Visualize loss graph
    plt.plot(training_times_under_3000, loss_under_3000_train)
    plt.plot(training_times_beyond_3000, loss_beyond_3000_train)
    plt.savefig("Output/trining_plot.png")  # Save the plot
    plt.show()

if __name__ == '__main__':

    # Checking the PyTorch Version
    print("PyTorch Version: ", torch.__version__)

    # Getting the device (in my case GPU with cuda 12.4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    count_image_dimensions('HipMRI_study_keras_slices_data/keras_slices_train')
    count_image_dimensions('HipMRI_study_keras_slices_data/keras_slices_test')
    count_image_dimensions('HipMRI_study_keras_slices_data/keras_slices_validate')

    dataloader = get_dataloader("HipMRI_study_keras_slices_data")

    # Verifying the dataset loader

    # Check the total number of images loaded
    dataset = dataloader.dataset
    print(f"Total number of images in the dataset: {len(dataset)}")

    # Fetch one sample and check its shape and type
    sample_image = dataset[0]  # Get the first image
    print(type(sample_image))
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample image data type: {sample_image.dtype}")

    visualize_samples(dataloader, num_samples=5)
    train()
