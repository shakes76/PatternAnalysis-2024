import torch
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from cleanfid import fid
from sklearn.preprocessing import StandardScaler
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
    
    def plot_losses(self, title, name):
        plt.figure(figsize=(10, 5))
        plt.plot(self.disc_losses, label="Discriminator Loss")
        plt.plot(self.gen_losses, label="Generator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(name)

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

    def predict_and_save(self, num_images, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
        latent_vectors = torch.randn(num_images, Z_DIM).to(DEVICE)

        self.gen.eval()

        with torch.no_grad():
            generated_images = self.gen(latent_vectors)

        generated_images = generated_images.cpu().detach()

        for i in range(num_images):

            save_path = os.path.join(save_dir, f"generated_image_{i + 1}.png")
 
            save_image(generated_images[i], save_path, normalize=True)
            print(f"Saved: {save_path}")

class FID_calculator:

    def __init__(self, AD_model_directory, NC_model_directory, number):
        self._ad_dir = AD_model_directory
        self._nc_dir = NC_model_directory
        self._number = number

    def move_real_images(self, target_dir, new_dir):
        files = os.listdir(target_dir)
    
        # Sort the files (optional, in case you want a specific order)
        files.sort()
        
        # Select the first X images
        files_to_move = files[:self._number]
        
        # Move the files
        for file_name in files_to_move:
            # Create full paths for the source and destination
            source_path = os.path.join(target_dir, file_name)
            destination_path = os.path.join(new_dir, file_name)
            shutil.copy(source_path, destination_path)
            print(f"Moved: {file_name}")

    def generate_fake_images(self,type,number,fake_dir):
        if type == "AD":
            model = self._ad_dir
        else:
            model = self._nc_dir

        model_loader = LoadModel(model)
        model_loader.load_model()
        model_loader.predict_and_save(number, fake_dir)

    def calculate_clean_fid(self,real_dir,fake_dir):
        return fid.calcualte(real_dir,fake_dir)
    
    


class LatentSpaceAnalyzer:
    def __init__(self, ad_model, nc_model, device='cuda'):
        self.ad_model = ad_model
        self.nc_model = nc_model
        self.device = device
        
    def generate_style_codes(self, num_samples=1000, batch_size=100):
        style_codes_ad = []
        style_codes_nc = []
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                # Use same latent vectors for both models for fair comparison
                z = torch.randn(current_batch_size, Z_DIM).to(self.device)
                
                # Get W space vectors from both models
                w_ad = self.ad_model.map(z).cpu().numpy()
                w_nc = self.nc_model.map(z).cpu().numpy()
                
                style_codes_ad.append(w_ad)
                style_codes_nc.append(w_nc)
        
        return np.vstack(style_codes_ad), np.vstack(style_codes_nc)

    def analyze_distributions(self, style_codes_ad, style_codes_nc):
        """Analyze statistical properties of the style codes"""
        print("AD Style Codes Statistics:")
        print(f"Mean: {np.mean(style_codes_ad):.4f}")
        print(f"Std: {np.std(style_codes_ad):.4f}")
        print(f"Min: {np.min(style_codes_ad):.4f}")
        print(f"Max: {np.max(style_codes_ad):.4f}")
        
        print("\nNC Style Codes Statistics:")
        print(f"Mean: {np.mean(style_codes_nc):.4f}")
        print(f"Std: {np.std(style_codes_nc):.4f}")
        print(f"Min: {np.min(style_codes_nc):.4f}")
        print(f"Max: {np.max(style_codes_nc):.4f}")
        
        # Calculate cosine similarity between random pairs
        from sklearn.metrics.pairwise import cosine_similarity
        random_indices = np.random.choice(len(style_codes_ad), size=1000)
        cos_sim = cosine_similarity(
            style_codes_ad[random_indices], 
            style_codes_nc[random_indices]
        )
        print(f"\nMean Cosine Similarity: {np.mean(cos_sim):.4f}")

    def plot_enhanced_tsne(self, num_samples=1000, perplexity=30, scale_data=True):
        """Plot TSNE with additional preprocessing and visualization enhancements"""
        # Generate style codes
        style_codes_ad, style_codes_nc = self.generate_style_codes(num_samples)
        
        # Combine data for scaling
        combined_data = np.vstack([style_codes_ad, style_codes_nc])
        
        # Scale data if requested
        if scale_data:
            scaler = StandardScaler()
            combined_data = scaler.fit_transform(combined_data)
            
        # Split back into AD and NC
        style_codes_ad = combined_data[:num_samples]
        style_codes_nc = combined_data[num_samples:]
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        combined_tsne = tsne.fit_transform(combined_data)
        
        # Split TSNE results
        tsne_ad = combined_tsne[:num_samples]
        tsne_nc = combined_tsne[num_samples:]
        
        # Create enhanced visualization
        plt.figure(figsize=(12, 8))
        
        # Plot with transparency and different markers
        plt.scatter(tsne_ad[:, 0], tsne_ad[:, 1], 
                   alpha=0.6, c='blue', label='AD', marker='o')
        plt.scatter(tsne_nc[:, 0], tsne_nc[:, 1], 
                   alpha=0.6, c='red', label='NC', marker='^')
        
        plt.title('t-SNE Visualization of StyleGAN Latent Space (W space)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        
        # Add density estimation
        from scipy.stats import gaussian_kde
        for data, color in [(tsne_ad, 'blue'), (tsne_nc, 'red')]:
            kde = gaussian_kde(data.T)
            x_range = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
            y_range = np.linspace(data[:, 1].min(), data[:, 1].max(), 100)
            xx, yy = np.meshgrid(x_range, y_range)
            positions = np.vstack([xx.ravel(), yy.ravel()])
            z = kde(positions)
            plt.contour(xx, yy, z.reshape(100, 100), levels=5, colors=color, alpha=0.3)
        
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return style_codes_ad, style_codes_nc

# Example usage:
# analyzer = LatentSpaceAnalyzer(ad_model.gen, nc_model.gen)
# style_codes_ad, style_codes_nc = analyzer.plot_enhanced_tsne(num_samples=2000, scale_data=True)
# analyzer.analyze_distributions(style_codes_ad, style_codes_nc)

