# Import dataset and network models
from dataset import *  
from modules import *  
# Import other necessary torch libraries 
import torch 
from tqdm import tqdm 
from torch import optim
import matplotlib.pyplot as plt

# Define constants and hyperparameters
DATASET                 = "./ADNI_AD_NC_2D/"  # Path to the dataset
DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available, otherwise use CPU
EPOCHS                  = 301 # Number of training epochs
LEARNING_RATE           = 1e-3  # Learning rate for optimization
BATCH_SIZE              = 32  # Batch size for training
LOG_RESOLUTION          = 7  # Logarithmic resolution used for 128*128 images
Z_DIM                   = 256  # Dimension of the latent space
W_DIM                   = 256  # Dimension of the mapping network output
LAMBDA_GP               = 10  # Weight for the gradient penalty term


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
    # Generate 'w' from random noise
    z = torch.randn(batch_size, W_DIM).to(DEVICE)
    w = mapping_network(z)
    return w[None, :, :].expand(LOG_RESOLUTION, -1, -1)

# Function to generate noise inputs for the generator
def get_noise(batch_size):
    # Generate noise inputs for the generator
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
):
    loop = tqdm(loader, leave=True)  # Create a tqdm progress bar for training iterations
    
    G_losses = []
    D_losses = []
    
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)  # Move real data to the specified device
        cur_batch_size = real.shape[0]

        w     = get_w(cur_batch_size)  # Generate 'w' from random noise
        noise = get_noise(cur_batch_size)  # Generate noise inputs for the generator
        with torch.cuda.amp.autocast():  # Use automatic mixed-precision (AMP) for faster training
            fake = gen(w, noise)  # Generate fake images
            critic_fake = critic(fake.detach())  # Get critic scores for fake images
            
            critic_real = critic(real)  # Get critic scores for real images
            gp = gradient_penalty(critic, real, fake, device=DEVICE)  # Compute gradient penalty
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))  # Critic loss
                + LAMBDA_GP * gp  # Gradient penalty term
                + (0.001 * torch.mean(critic_real ** 2))  # Regularization term
            )
        D_losses.append(loss_critic.item())

        critic.zero_grad()  # Reset gradients for the critic
        loss_critic.backward()  # Backpropagate the critic loss
        opt_critic.step()  # Update critic's weights

        gen_fake = critic(fake)  # Get critic scores for fake images
        loss_gen = -torch.mean(gen_fake)  # Generator loss

        if batch_idx % 16 == 0:  # Apply path length penalty every 16 batches
            plp = path_length_penalty(w, fake)
            if not torch.isnan(plp):
                loss_gen = loss_gen + plp  # Update generator loss with path length penalty
        G_losses.append(loss_gen.item())
        mapping_network.zero_grad()  # Reset gradients for the mapping network
        gen.zero_grad()  # Reset gradients for the generator
        loss_gen.backward()  # Backpropagate the generator loss
        opt_gen.step()  # Update generator's weights
        opt_mapping_network.step()  # Update mapping network's weights

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )
        
        
    
    return (D_losses, G_losses)
    

# Initialize the mapping network, generator, and critic on the specified device
mapping_network     = MappingNetwork(Z_DIM, W_DIM).to(DEVICE)  # Initialize mapping network
gen                 = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)  # Initialize generator
critic              = Discriminator(LOG_RESOLUTION).to(DEVICE)  # Initialize critic

loader = get_data_loader(DATASET, LOG_RESOLUTION, BATCH_SIZE)

gen                 = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)

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
    G_Losses , D_Losses = train_fn(
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
plt.ylabel("Loss")
plt.legend()
plt.savefig('gen_loss.png')

plt.figure(figsize=(10,5))
plt.title("Discriminator Loss During Training")
plt.plot(Total_D_Losses, label="D", color="orange")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('dis_loss.png')