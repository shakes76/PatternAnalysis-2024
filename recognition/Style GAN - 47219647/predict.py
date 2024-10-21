from torch.utils.data import DataLoader
import torch
from sklearn.manifold import TSNE
from modules import *

class LoadModel:
    def __init__(self, file_path="model_checkpoint.pth"):
        self.checkpoint = torch.load(file_path)
        self.gen = None
        self.disc = None
        self.opt_gen = None
        self.opt_disc = None
        self.epoch = None
        self.step = None
        self.disc_losses = []
        self.gen_losses = []

    def load_plotting(self):
        self.disc_losses = self.checkpoint.get("disc_losses", [])
        self.gen_losses = self.checkpoint.get("gen_losses", [])
        
    # Internal function to initialize models and load states
    def load_model(self):
        # # Initialize generator and discriminator models
        self.gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
        self.disc = Discriminator(IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
        
        # Initialize optimizers
        self.opt_gen = optim.Adam([
            {"params": [param for name, param in self.gen.named_parameters() if "map" not in name]},
            {"params": self.gen.map.parameters(), "lr": 1e-5}
        ])
        self.opt_disc = optim.Adam(self.disc.parameters())
        
        # Load model states
        self.gen.load_state_dict(self.checkpoint["generator_state_dict"])
        self.disc.load_state_dict(self.checkpoint["discriminator_state_dict"])
        self.opt_gen.load_state_dict(self.checkpoint["opt_gen_state_dict"])
        self.opt_disc.load_state_dict(self.checkpoint["opt_disc_state_dict"])

        # Load additional data
        self.epoch = self.checkpoint["epoch"]
        self.step = self.checkpoint["step"]
        self.disc_losses = self.checkpoint.get("disc_losses", [])
        self.gen_losses = self.checkpoint.get("gen_losses", [])
        
    # Getter methods for each component
    def get_generator(self):
        return self.gen
    
    def get_discriminator(self):
        return self.disc
    
    def get_opt_gen(self):
        return self.opt_gen
    
    def get_opt_disc(self):
        return self.opt_disc
    
    def get_epoch(self):
        return self.epoch
    
    def get_step(self):
        return self.step
    
    def get_disc_losses(self):
        return self.disc_losses
    
    def get_gen_losses(self):
        return self.gen_losses
    
    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.disc_losses, label="Discriminator Loss")
        plt.plot(self.gen_losses, label="Generator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Discriminator and Generator Losses Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict(self, num_images):
        latent_vectors = torch.randn(num_images, Z_DIM).to(DEVICE)

        self.gen.eval()

        with torch.no_grad():

            generated_images = self.gen(latent_vectors)

        generated_images = generated_images.cpu().detach()
        plt.figure(figsize=(10, 10))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(generated_images[i].permute(1, 2, 0).squeeze(), cmap='gray')
            plt.axis('off')
        plt.show()

    
    def plot_tsne_style_space(self, num_samples):
        # Generate random latent vectors (Z space)
        latent_vectors = torch.randn(num_samples, Z_DIM).to(DEVICE)
        
        # Pass latent vectors through the generator's mapping network to get style codes (W space)
        with torch.no_grad():
            style_codes = self.gen.map(latent_vectors)

        # Apply t-SNE to the style codes (W space)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        style_codes_2d = tsne.fit_transform(style_codes.cpu().numpy())

        # Plot the t-SNE embedding of the style codes (W space)
        plt.figure(figsize=(10, 8))
        plt.scatter(style_codes_2d[:, 0], style_codes_2d[:, 1], s=5, cmap='Spectral')
        plt.title('t-SNE embedding of StyleGAN Latent Space (W space)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.colorbar()
        plt.show()


