# In your train.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchmetrics.functional.image \
    import structural_similarity_index_measure as ssim
import numpy as np
import matplotlib.pyplot as plt
import os

from dataset import HipMRILoader
import modules

class ValidationLossEarlyStopping:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience  # number of times to allow for no improvement before stopping the execution
        self.min_delta = min_delta  # the minimum change to be counted as improvement
        self.counter = 0  # count the number of times the validation accuracy not improving
        self.min_validation_loss = np.inf

    # return True when validation loss is not decreased by the `min_delta` for `patience` times 
    def early_stop_check(self, validation_loss):
        if ((validation_loss + self.min_delta) < self.min_validation_loss):
            self.min_validation_loss = validation_loss
            self.counter = 0  # reset the counter if validation loss decreased at least by min_delta
        elif ((validation_loss+self.min_delta) > self.min_validation_loss):
            self.counter += 1 # increase the counter if validation loss is not decreased by the min_delta
            if self.counter >= self.patience:
                return True
        return False


# # Hyperparameters
# num_epochs = 200
# batch_size = 32
# lr = 0.0002
# num_hiddens = 128
# num_residual_hiddens = 32
# num_channels = 1
# num_embeddings = 512
# dim_embedding = 64
# beta = 0.25

num_epochs = 150
batch_size = 16
lr = 0.003
num_hiddens = 128
num_residual_hiddens = 32
num_channels = 1
num_embeddings = 512
dim_embedding = 64
beta = 0.25

# Save directory
save_dir = f'test'

# Define the save directory and ensure it exists
model_dir = f'saved_model/test.pth'

def train_model(
        save_dir=save_dir | None, 
        model_dir=model_dir | None, 
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        num_hiddens=num_hiddens,
        num_residual_hiddens=num_residual_hiddens,
        num_channels=num_channels,
        num_embeddings=num_embeddings,
        dim_embedding=dim_embedding,
        beta=beta
        ):

    # Hyperparameters
    num_epochs = num_epochs
    batch_size = batch_size
    lr = lr
    num_hiddens = num_hiddens
    num_residual_hiddens = num_residual_hiddens
    num_channels = num_channels
    num_embeddings = num_embeddings
    dim_embedding = dim_embedding
    beta = beta

    print("==== Paramaters ====")
    print(f"Max Epochs: ", num_epochs)
    print(f"Batch size: ", batch_size)
    print(f"lr: ", lr)
    print(f"num_hiddens: ", num_hiddens)
    print(f"num_residual_hiddens: ", num_residual_hiddens)
    print(f"num_channels", num_channels)
    print(f"num_embeddings: ", num_embeddings)
    print(f"dim_embedding: ", dim_embedding)
    print(f"beta: ", beta)
    print()

    # Save directory
    if type(save_dir) == str:
        os.makedirs(save_dir, exist_ok=True)

    # Model directory
    if type(model_dir) == str:
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)

    # Configure Pytorch
    seed = 42
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    print()

    # Directories for datasets
    train_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'
    test_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'
    validate_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'

    # Define your transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  
        # transforms.Normalize((0.5,), (0.5,)),
        transforms.Normalize((0,), (1,)) 
    ])

    # Get loaders (variance == 5)
    train_loader, validate_loader, data_variance = HipMRILoader(
        train_dir, validate_dir, test_dir,
        batch_size=batch_size, transform=transform
        ).get_loaders()
    # print('Variance: ', data_variance)

    # Assuming you have a DataLoader defined as train_loader
    # dataset_size = len(train_loader.dataset)  # Size of the training dataset
    # iterations_per_epoch = len(train_loader)   # Number of batches per epoch

    # print(f"Training dataset size: {dataset_size}") #11460
    # print(f"Iterations per epoch: {iterations_per_epoch}")

    # Create model
    model = modules.VQVAE(
        num_channels=num_channels,
        num_hiddens=num_hiddens,
        num_residual_hiddens=num_hiddens,
        num_embeddings=num_embeddings,
        dim_embedding=dim_embedding,
        beta=beta)

    model = model.to(device)

    # Set optimiser
    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr,
        amsgrad=False)

    # # Define the scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # # Piecewise Linear Schedule. Helps the optimiser frind the appropriate local minima without overshoooting. 
    # # CyclicLR explores a range of learning rates in the early part of training. lr fluctuates between base_lr and max_lr
    # sched_linear_1 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0004, max_lr=lr, step_size_up=715, step_size_down=715, mode="triangular", verbose=False)
    # # LinearLR reduces the learning rate steadily in the latter part of training to help the model converge more smoothly.
    # # Slowly decrease lr from start_factor to end_factor. Can make more fine adjstments toeards end of training. 
    # sched_linear_2 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.0004/lr, end_factor=0.0004/5, verbose=False)
    # # SequentialLR combines these two schedules, making the learning rate change in a piecewise linear manner. This combination helps with faster convergence while maintaining stability at the end of training.
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_linear_1, sched_linear_2], milestones=[40])

    # early_stopper = ValidationLossEarlyStopping(40, 0.01)

    # Training mectrics
    epoch_training_output_loss = []
    epoch_training_vq_loss = []

    epoch_validation_output_loss = []
    epoch_validation_vq_loss = []
    epoch_ssim = []


    # Training loop
    for epoch in range(num_epochs):
        model.train()
        training_output_error = []
        training_vq_error = []

        for i, training_images in enumerate(train_loader):
            training_input_images = training_images.to(device)

            optimizer.zero_grad()
            vq_loss, training_output_images = model(training_input_images)

            # Calculate reconstruction loss
            output_loss = F.mse_loss(training_output_images, training_input_images)
            loss = output_loss + vq_loss
            loss.backward()

            optimizer.step()

            training_output_error.append(output_loss.item())
            training_vq_error.append(vq_loss.item())
        
        # Calculate and store average training losses
        training_output_loss = np.mean(training_output_error)
        epoch_training_output_loss.append(training_output_loss)
        training_vq_loss = np.mean(training_vq_error)
        epoch_training_vq_loss.append(training_vq_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training output loss: {training_output_loss:.5f}, Training VQ loss: {training_vq_loss}')

        # Evaluate on the validation dataset
        model.eval()
        validation_output_error = []
        validation_vq_error = []
        validation_ssim = []

        with torch.no_grad():
            for j, validation_images in enumerate(validate_loader):
                validation_input_images = validation_images.to(device)
                validation_vq_loss, validation_output_images = model(validation_input_images)

                # Reshape images for SSIM calculation
                real_image = validation_input_images.view(-1, 1, 256, 128).detach().to(device)
                decoded_image = validation_output_images.view(-1, 1, 256, 128).detach().to(device)
                
                # Calculate SSIM and store it
                similarity = ssim(decoded_image, real_image, data_range=1.0).item()
                validation_ssim.append(similarity)

                # Calculate output loss for validation
                validation_output_loss = F.mse_loss(validation_output_images, validation_input_images)
                validation_output_error.append(validation_output_loss.item())
                validation_vq_error.append(validation_vq_loss.item())

        # Average validation loss and SSIM
        average_validation_loss = np.mean(validation_output_error)
        epoch_validation_output_loss.append(average_validation_loss)
        average_validation_vq_loss = np.mean(validation_vq_error)
        epoch_validation_vq_loss.append(average_validation_vq_loss)
        average_ssim = np.mean(validation_ssim)
        epoch_ssim.append(average_ssim)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation output loss: {average_validation_loss:.5f}, Validation VQ loss: {average_validation_vq_loss}, Average SSIM: {average_ssim:.5f}')

        if (epoch + 1) % 10 == 0:
            # Number of images to save
            num_images_to_save = 4  # Adjust this for the number of real and decoded images

            # Convert tensors to numpy arrays
            real_images = validation_input_images.cpu().numpy()
            decoded_images = validation_output_images.cpu().numpy()  

            # Create a new figure for saving real and decoded images side by side
            fig, axes = plt.subplots(num_images_to_save, 2, figsize=(8, num_images_to_save * 4))  # 2 columns

            # Plot real and decoded images
            for k in range(num_images_to_save):
                # Plot real images
                axes[k, 0].imshow(real_images[k, 0], cmap='gray')
                axes[k, 0].set_title('Real Image')
                axes[k, 0].axis('off')

                # Plot decoded images
                axes[k, 1].imshow(decoded_images[k, 0], cmap='gray')
                axes[k, 1].set_title('Decoded Image')
                axes[k, 1].axis('off')

            # Add a title for the entire figure with the epoch number
            fig.suptitle(f'Epoch {epoch + 1}', fontsize=16)

            # Adjust layout for better spacing
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the title

            # Save the figure as a single PNG file with the epoch number in the filename
            plt.savefig(os.path.join(save_dir, f'real_and_decoded_images_epoch_{epoch + 1}.png'))
            plt.close()

        # if early_stopper.early_stop_check(average_validation_loss):
        #     print(f"Stopped early at Epoch: {epoch +1}")
        #     break
        
        # scheduler.step()
        # print(f"Epoch [{epoch+1}/{num_epochs}], Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        print()

    if type(save_dir) == str:
        epochs = range(1, len(epoch_training_output_loss) + 1)

        # 1. Plot Training Output Loss vs. Validation Output Loss
        plt.figure(figsize=(12, 6))  # Width of normal page
        plt.plot(epochs, epoch_training_output_loss, label='Training Output Loss')
        plt.plot(epochs, epoch_validation_output_loss, label='Validation Output Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Output Loss')
        plt.title('Training and Validation Output Loss per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'output_loss_per_epoch.png'))
        plt.close()

        # 2. Plot Training VQ Loss vs. Validation VQ Loss
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, epoch_training_vq_loss, label='Training VQ Loss')
        plt.plot(epochs, epoch_validation_vq_loss, label='Validation VQ Loss')
        plt.xlabel('Epochs')
        plt.ylabel('VQ Loss')
        plt.title('Training and Validation VQ Loss per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'vq_loss_per_epoch.png'))
        plt.close()

        # 3. Plot Combined Training Loss vs. Validation Loss
        combined_training_loss = [x + y for x, y in zip(epoch_training_output_loss, epoch_training_vq_loss)]
        combined_validation_loss = [x + y for x, y in zip(epoch_validation_output_loss, epoch_validation_vq_loss)]

        plt.figure(figsize=(12, 6))
        plt.plot(epochs, combined_training_loss, label='Combined Training Loss')
        plt.plot(epochs, combined_validation_loss, label='Combined Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Combined Loss')
        plt.title('Combined Training and Validation Loss per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'combined_loss_per_epoch.png'))
        plt.close()

        # 4. Plot SSIM per Epoch
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, epoch_ssim, label='SSIM')
        plt.xlabel('Epochs')
        plt.ylabel('SSIM')
        plt.title('SSIM per Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'ssim_per_epoch.png'))
        plt.close()

    if type(model_dir) == str:
        # Save the model state_dict
        torch.save(model.state_dict(), model_dir)

    return model


if (__name__ == "__main__"):
    model = train_model(save_dir=save_dir, model_dir=model_dir)

