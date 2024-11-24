"""
Author: Ewan Stanich s4742842

This file contains the training, validation and testing functionality to train a VQVAE.
"""
import math
from dataset import load_data
import torch
from tqdm import tqdm
import time
import utils
from modules import VQVAE
from predict import generate_samples

# Hyperparameters
IS_TESTING = False
HIDDEN_DIM = 128
RES_HIDDEN_DIM = 32
N_RES_LAYERS = 5
N_EMBEDDINGS = 512
EMBEDDING_DIM = 64
LEARNING_RATE = 3e-4
N_EPOCHS = 10 if IS_TESTING else 150
BATCH_SIZE = 16
BETA = 0.25 # As per the original paper

# Folder Locations
MODEL_LOCATION = "./models/"
IMAGE_LOCATION = "./epoch_reconstructions/"
OUTPUT_LOCATION = "./outputs/"

# Metrics
ssim_scores = []
train_losses = []
best_epoch = 0

# Initialize model and optimisers
utils.folder_check(output_loc=OUTPUT_LOCATION, model_loc=MODEL_LOCATION, image_loc=IMAGE_LOCATION) # Ensure the output directories exist and are empty from prior runs
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")  # Initialise the device
train_loader, test_loader, val_loader = load_data(testing=IS_TESTING, batch_size=BATCH_SIZE) # Initialise the data loader
model = VQVAE(HIDDEN_DIM,
              RES_HIDDEN_DIM,
              N_RES_LAYERS,
              N_EMBEDDINGS,
              EMBEDDING_DIM,
              BETA).to(device)  # Initialise the model

# We opt for the Adam optimiser in this model
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)
# MSE as require for two of the components of our loss functions
criterion = torch.nn.MSELoss()


def train_model():
    """
        Trains the VQVAE model with training data from a data loader that runs in batches over a specified number of
        epochs. Losses are calculated and backpropagated to update the model weights. The model is validated after each
        epoch and the best model is saved based on the SSIM score. Images are generated every 10 epochs for progressive
        comparison.
    """

    for epoch_idx in range(N_EPOCHS):
        print("Training")

        # Initialise/reset epoch metrics
        epoch_loss = 0
        epoch_start = time.time()

        model.train()  # Set model to training mode

        for batch, im in enumerate(tqdm(train_loader)):
            start_time = time.time()  # For batch metrics

            im = im.float().unsqueeze(1).to(device)  # Ensures image is correct shape for model
            opt.zero_grad()  # Reset gradients in the optimiser

            decoded_output, embedding_loss, encoded_output, quantised_output = model(im)
            recon_loss = criterion(decoded_output, im)  # Loss between encoder and decoder
            loss = recon_loss + embedding_loss  # combines all the loss for a single metric
            loss.backward()  # backpropagate the loss
            opt.step()  # step the optimiser

            epoch_loss += loss.item()  # increase the total epoch loss

            # Print progress every 64 batches in each epoch
            if (batch + 1) % 64 == 0:
                print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                    batch * len(im), len(train_loader.dataset),
                    50 * batch / len(train_loader),
                    epoch_loss / batch,
                    time.time() - start_time
                ))

        # Calculate the average loss for the epoch, print and store
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(math.log(avg_epoch_loss))
        print('Finished epoch {} in time: {} with loss:'.format(
            epoch_idx + 1, time.time() - epoch_start, avg_epoch_loss))

        # Validate the model after each epoch
        ssim_score = validate_model(epoch_idx + 1)
        
        # Generate samples every 2 epochs
        if (epoch_idx + 1) % 2 == 0:
            print("Generating Epoch Image")
            generate_samples(test_loader, model, MODEL_LOCATION, ssim_score, IMAGE_LOCATION, epoch_idx + 1)
        
        utils.plot_results(train_losses, ssim_scores, epoch_idx + 1, OUTPUT_LOCATION)

    print('Done Training...')


def validate_model(epoch):
    """
        Validates the model based on the calculated SSIM score, ensuring the best model is saved based on the highest
        SSIM score, as training can be volatile and the best model may not be the last epoch.
    """
    print("Validating")
    global best_epoch  # Ensure we have access to the best epoch
    model.eval()  # Set model to evaluation model so it does not train while we test it
    total_ssim = 0  # Initialise epoch SSIM

    with torch.no_grad():  # We don't want to use the gradient for validation
        for batch, im in enumerate(val_loader):
            im = im.float().unsqueeze(1).to(device)  # Ensures image is correct shape for model

            decoded_output, _, _, _ = model(im)  # We only want the image, which is the decoded output

            total_ssim += utils.calc_ssim(decoded_output, im)  # Calculate the SSIM score and add to total

    epoch_ssim_score = total_ssim / (batch + 1)  # Find average SSIM score of all batches
    ssim_scores.append(epoch_ssim_score)  # Store the SSIM score for this epoch

    # Save the model if it is the best model based on the SSIM score
    if epoch_ssim_score == max(ssim_scores):
        torch.save(model.state_dict(), MODEL_LOCATION + f'checkpoint_epoch{epoch}_vqvae.pt')
        best_epoch = epoch
        print(f"Achieved an SSIM score of {epoch_ssim_score}, NEW BEST! saving model")
    else:
        print(f"Achieved an SSIM score of {epoch_ssim_score}")
    
    return epoch_ssim_score


def test():
    """
        Tests the final model, which achieved the best SSIM score on the validation set. We wish to see how this mdoel
        will do against unseen data in the test set and thus this is the final gauge of how well the model will perform.

        Output:
            test_ssim: The average SSIM score of the test set over the batches of the test dataset
    """
    print("Testing")
    model.load_state_dict(torch.load(MODEL_LOCATION + f'checkpoint_epoch{best_epoch}_vqvae.pt'))
    torch.save(model.state_dict(), OUTPUT_LOCATION + f'final_vqvae.pt')
    model.eval()
    total_ssim = 0

    with torch.no_grad():
        for batch, im in enumerate(test_loader):
            im = im.float().unsqueeze(1).to(device)  # Ensures image is correct shape for model

            decoded_output, _, _, _ = model(im)  # We only want the image, which is the decoded output

            total_ssim += utils.calc_ssim(decoded_output, im)  # Calculate the SSIM score and add to total

    final_ssim = total_ssim / (batch + 1)  # Find average SSIM score of all batches
    return final_ssim


if __name__ == "__main__":
    start = time.time()
    train_model()
    print(f"Took {(time.time() - start) / 60} minutes to train")
    test_ssim = test()
    print(f"Test SSIM achieved as {test_ssim}")
    utils.plot_results(train_losses, ssim_scores, N_EPOCHS, OUTPUT_LOCATION)
    generate_samples(test_loader, model, OUTPUT_LOCATION, test_ssim, IMAGE_LOCATION)
