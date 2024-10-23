# Import dataset and network models
from dataset import ImageDataset  # Assuming this is the class you defined
from modules import *  
import torch 
from tqdm import tqdm 
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

# Define constants and hyperparameters
DATASET                 = "./AD/train"  # Path to the training dataset
DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise use CPU
EPOCHS                  = 100  # Number of training epochs
LEARNING_RATE           = 1e-3  # Learning rate for optimization
BATCH_SIZE              = 32  # Batch size for training
LOG_RESOLUTION          = 7  # Logarithmic resolution used for 128*128 images
Z_DIM                   = 256  # Dimension of the latent space
W_DIM                   = 256  # Dimension of the mapping network output
LAMBDA_GP               = 15  # Weight for the gradient penalty term

# Define transformations for the dataset (resize to 128x128, normalize to [-1, 1])
image_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for 3 channels
])

# Create the dataset and DataLoader
train_dataset = ImageDataset(image_dir=DATASET, transform=image_transforms)
loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# Function to compute the gradient penalty for the discriminator
def gradient_penalty(critic, real, fake, device="cpu"):
    # Compute gradient penalty for the critic
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate discriminator scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

# Function to generate latent vectors 'w' from random noise
def get_w(batch_size):
    z = torch.randn(batch_size, W_DIM).to(DEVICE)
    w = mapping_network(z)
    return w[None, :, :].expand(LOG_RESOLUTION, -1, -1)

# Function to generate noise inputs for the generator
def get_noise(batch_size):
    noise = []
    resolution = 4

    for i in range(LOG_RESOLUTION):
        if i == 0:
            n1 = None
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)
        noise.append((n1, n2))
        resolution *= 2

    return noise

# Training function for the discriminator and generator
def train_fn(
    critic,
    gen,
    path_length_penalty,
    loader,
    opt_critic,
    opt_gen,
    opt_mapping_network,
    gamma=20,  # R1 regularization weight, typically 10
):
    loop = tqdm(loader, leave=True)  # Create a tqdm progress bar for training iterations

    scaler = torch.amp.GradScaler('cuda')  # GradScaler for AMP

    G_losses = []
    D_losses = []
    
    for batch_idx, real in enumerate(loop):
        real = real.to(DEVICE)  # Move real data to the specified device
        cur_batch_size = real.shape[0]

        # Ensure real images have requires_grad=True for R1 regularization
        real.requires_grad_(True)

        w = get_w(cur_batch_size)  # Generate 'w' from random noise
        noise = get_noise(cur_batch_size)  # Generate noise inputs for the generator

        # Using updated AMP handling for faster training
        with torch.amp.autocast('cuda'):  # Use automatic mixed-precision (AMP)
            fake = gen(w, noise)  # Generate fake images
            critic_fake = critic(fake.detach())  # Get critic scores for fake images
            critic_real = critic(real)  # Get critic scores for real images

            # R1 regularization for real data (StyleGAN specific)
            r1_penalty = torch.autograd.grad(
                outputs=critic_real.sum(),
                inputs=real,
                create_graph=True
            )[0].pow(2).view(real.size(0), -1).sum(1).mean()

            # Critic loss calculation with R1 regularization
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))  # Standard loss
                + (gamma / 2) * r1_penalty  # R1 penalty
                + (0.001 * torch.mean(critic_real ** 2))  # Regularization term
            )

        print(f"critic_real: {critic_real.mean().item()}, critic_fake: {critic_fake.mean().item()}")
        print(f"r1_penalty: {r1_penalty.item()}, loss_critic: {loss_critic.item()}")

        # Store the critic loss
        D_losses.append(loss_critic.item())

        # Update critic
        critic.zero_grad()  # Reset gradients for the critic
        scaler.scale(loss_critic).backward()  # Use scaled backward for AMP
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        scaler.step(opt_critic)  # Use scaled step
        scaler.update()  # Update the scaler

        # Generator loss calculation and update
        with torch.amp.autocast('cuda'):
            gen_fake = critic(fake)  # Get critic scores for fake images
            loss_gen = -torch.mean(gen_fake)  # Generator loss

        # Apply path length penalty every 16 batches
        if batch_idx % 16 == 0:
            plp = path_length_penalty(w, fake)
            loss_gen = loss_gen + plp  # Update generator loss with path length penalty

        G_losses.append(loss_gen.item())  # Store the generator loss

        # Reset gradients for the mapping network and generator
        mapping_network.zero_grad()  # Reset gradients for the mapping network
        gen.zero_grad()  # Reset gradients for the generator
        scaler.scale(loss_gen).backward()  # Backpropagate the generator loss with AMP
        scaler.step(opt_gen)  # Use scaled step for generator
        scaler.update()  # Update the scaler
        opt_mapping_network.step()  # Update mapping network's weights

        # Optionally, log progress
        loop.set_postfix(
            r1_penalty=r1_penalty.item(),
            loss_critic=loss_critic.item(),
        )
    
    return (D_losses, G_losses)



# Initialize the mapping network, generator, and critic on the specified device
mapping_network     = MappingNetwork(Z_DIM, W_DIM).to(DEVICE)  # Initialize mapping network
gen                 = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)  # Initialize generator
critic              = Discriminator(LOG_RESOLUTION).to(DEVICE)  # Initialize critic

path_length_penalty = PathLengthPenalty(0.99).to(DEVICE)

opt_gen             = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_critic          = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

gen.train()
critic.train()
mapping_network.train()

Total_G_Losses = []
Total_D_Losses = []

for epoch in range(EPOCHS):
    G_Losses, D_Losses = train_fn(
        critic,
        gen,
        path_length_penalty,
        loader,
        opt_critic,
        opt_gen,
        opt_mapping_network,
    )
    
    Total_G_Losses.extend(G_Losses)
    Total_D_Losses.extend(D_Losses)
    
    if epoch % 20 == 0:
        torch.save(gen.state_dict(), f'generator_epoch{epoch}.pt')

plt.figure(figsize=(10,5))
plt.title("Generator Loss During Training")
plt.plot(Total_G_Losses, label="G", color="blue")
plt.xlabel("iterations")
