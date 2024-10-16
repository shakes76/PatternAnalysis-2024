import torch
import matplotlib.pyplot as plt
import modules, dataset

def add_noise(image, beta, t):
    """
    Add noise to image
    :param image: input image
    :param beta: diffusion coefficient
    :param t: timestep
    :return:
    """
    noise = torch.randn_like(image)  # Generate random noise
    noisy_image = image * torch.sqrt(beta[t]) + noise * torch.sqrt(1 - beta[t])  # Add noise
    return noisy_image

def plot_forward_process(data_loader,  timesteps=5):
    """
    Visualise forward process
    :param data_loader: training dataloader
    :param timesteps: number of timesteps
    :return:
    """
    # Get a batch of images from the DataLoader
    images, _ = next(iter(data_loader))  # Assuming the DataLoader returns (images, labels)

    # Select the first image from the batch
    image = images[0]  # Choose any index or a specific image
    plt.figure(figsize=(12, 6))

    # Plot the original image
    plt.subplot(1, timesteps + 1, 1)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.title("Original Image")
    plt.axis('off')

    # Plot noisy images at various timesteps
    for t in range(timesteps):
        # Create gradual noise by interpolating
        noise = torch.randn_like(image)  # Generate random noise
        noisy_image = (1 - (t / (timesteps - 1))) * image + (t / (timesteps - 1)) * noise
        plt.subplot(1, timesteps + 1, t + 2)
        plt.imshow(noisy_image.permute(1, 2, 0).cpu().numpy())
        plt.title(f"Timestep {t + 1}")
        plt.axis('off')

    plt.show()

ddpm_scheduler = modules.DiffusionScheduler(num_time_steps=1000)
plot_forward_process(dataset.dataloader)
