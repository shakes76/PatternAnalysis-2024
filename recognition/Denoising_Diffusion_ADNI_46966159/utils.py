import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from timm.utils import ModelEmaV3
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

def tsne():
    """
    t-sne plot for visudalisation.
    Load the model and perform dimension reduction
    :return: t-SNE plot
    """
    # load model parameters
    dataloader = dataset.dataloader
    checkpoint = torch.load("ddpm_checkpoint")
    model = modules.UNET()
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=0.9999)
    ema.load_state_dict(checkpoint['ema'])

    # store features and ground truth labels
    features = []
    labels = []

    with torch.no_grad():  # Disable gradient calculation
        for images, label in dataloader:
            output = model(images).view(images.size(0), -1)  # Flatten output
            features.append(output)
            labels.append(label)

    features = torch.cat(features).numpy()  # Convert to NumPy
    labels = torch.cat(labels).numpy()  # Convert to NumPy


    # dimension reduction to 2 dims
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    # plot tsne
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels)
    plt.colorbar(scatter, ticks=[0, 1], label='Class')
    plt.title('t-SNE visualization of ADNI Images')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()
    plt.savefig("tsne.png")

ddpm_scheduler = modules.DiffusionScheduler(num_time_steps=1000)
plot_forward_process(dataset.dataloader)
tsne()
