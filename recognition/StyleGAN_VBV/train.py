import os
import torch
import torch.optim as optim
from tqdm import tqdm
from modules import Generator, Discriminator
from dataset.py import get_loader 

# Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Z_DIM = 512  # Latent space dimension
W_DIM = 512  # Intermediate latent space dimension
IN_CHANNELS = 512  # Initial input channels
CHANNELS_IMG = 3  # Number of output image channels
LR = 0.001  # Learning rate
BATCH_SIZE = 16  # Batch size
NUM_EPOCHS = 100  # Number of epochs for training

# Function to generate examples
def generate_examples(gen, n=100):
    gen.eval()
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, Z_DIM).to(DEVICE)
            img = gen(noise, 1.0, 0)  # Alpha=1.0, steps=0 for full-size image
            if not os.path.exists('saved_examples'):
                os.makedirs('saved_examples')
            save_image(img * 0.5 + 0.5, f"saved_examples/img_{i}.png")
    gen.train()

# Training function
def train_fn(critic, gen, loader, opt_critic, opt_gen):
    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]
        noise = torch.randn(cur_batch_size, Z_DIM).to(DEVICE)
        fake = gen(noise, 1.0, 0)  # Generate fake images

        # Critic loss
        loss_critic = -torch.mean(critic(real)) + torch.mean(critic(fake.detach()))
        
        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        # Generator loss
        loss_gen = -torch.mean(critic(fake))
        
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        loop.set_postfix(loss_critic=loss_critic.item(), loss_gen=loss_gen.item())

# Main training loop
def main():
    gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG).to(DEVICE)
    critic = Discriminator(IN_CHANNELS, CHANNELS_IMG).to(DEVICE)
    
    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=LR, betas=(0.0, 0.99))

    loader, dataset = get_loader(BATCH_SIZE)
    
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}]')
        train_fn(critic, gen, loader, opt_critic, opt_gen)

    generate_examples(gen)

if __name__ == '__main__':
    main()
