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
    """
    Class that is responsible for loading saved model and ploting basic stats
    from the saved models
    """
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
        """
        Weaker method used only when plotting is needed
        """
        self.disc_losses = self.checkpoint.get("disc_losses", [])
        self.gen_losses = self.checkpoint.get("gen_losses", [])
        

    def load_model(self):
        """
        Full loading function to initialize models and load states
        """
        # Initialize generator and discriminator 
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
        """
        Plots the losses based on what the model saved
        """
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
        """
        Uses the generator to predict and product images for quality checking.
        """
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
    """
    This class is responsible for calculating the FID (Fréchet Inception Distance).
    This is an important metric used to bechmark model based on the diversity of 
    images they produce.
    This class also includes important methods to calculate the FID.
    """

    def __init__(self, AD_model_directory, NC_model_directory, number):
        self._ad_dir = AD_model_directory
        self._nc_dir = NC_model_directory
        self._number = number

    def move_real_images(self, target_dir, new_dir):
        """
        This function samples the real image data base and moved it to a seperate
        folder.
        """
        files = os.listdir(target_dir)

        files_to_move = files[:self._number]
        
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
    """
    This class is responsible for plotting the TSNE for the latent spaces for the
    two models which were trained on AD and NC respectively. After creating the 
    plots there should a seperation between the 2 clusters, illustaraing how 
    StyleGAN and learnt to distinguish between these 2 styles.
    """
    
    def __init__(self, ad_model, nc_model, num_samples = 1000):
        #Defines models and number of samples
        self.ad_model = ad_model
        self.nc_model = nc_model
        self.num_samples = num_samples
        self.device = DEVICE
        
    def generate_style_codes(self,  batch_size=100):
        """
        Geneartes the styles codes for the given latens spaces
        """
        #Style container that stores the style code
        style_codes_ad = []
        style_codes_nc = []
        
        #Creates the Style codes in batches
        with torch.no_grad():
            for i in range(0, self.num_samples, batch_size):
                current_batch_size = min(batch_size, self.num_samples - i)
                
                z = torch.randn(current_batch_size, Z_DIM).to(self.device)
                
                w_ad = self.ad_model.map(z).cpu().numpy()
                w_nc = self.nc_model.map(z).cpu().numpy()
                
                style_codes_ad.append(w_ad)
                style_codes_nc.append(w_nc)

        #Restructure the array to remove batches 
        return np.vstack(style_codes_ad), np.vstack(style_codes_nc)

    def checks_cosinsimilarity(self, style_codes_ad, style_codes_nc):
        """
        Prints the important statistics for the style codes of each model
        """
        
        from sklearn.metrics.pairwise import cosine_similarity
        random_indices = np.random.choice(len(style_codes_ad), size=1000)
        cos_sim = cosine_similarity(
            style_codes_ad[random_indices], 
            style_codes_nc[random_indices]
        )
        print(f"\nMean Cosine Similarity: {np.mean(cos_sim):.4f}")

    def plot_tsne(self, perplexity=30):
        """
        Plot TSNE using the AD and NC model latent spaces
        """
        # Generate style codes
        style_codes_ad, style_codes_nc = self.generate_style_codes(self.num_samples)
        
        # Combine data for scaling
        combined_data = np.vstack([style_codes_ad, style_codes_nc])

        scaler = StandardScaler()
        style_codes_ad = scaler.fit_transform(combined_data)
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        combined_tsne = tsne.fit_transform(combined_data)
        
        # Split TSNE results
        tsne_ad = combined_tsne[:self.num_samples]
        tsne_nc = combined_tsne[self.num_samples:]
        
        plt.figure(figsize=(12, 8))
        plt.scatter(tsne_ad[:, 0], tsne_ad[:, 1], 
                   alpha=0.6, c='blue', label='AD', marker='o')
        plt.scatter(tsne_nc[:, 0], tsne_nc[:, 1], 
                   alpha=0.6, c='red', label='NC', marker='o')
        plt.title('t-SNE Visualization of StyleGAN Latent Space (W space)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()        
        plt.grid(True, alpha=0.3)
        plt.show()
        plt.savefig("2D TSNE plot")
        
        # Split back into AD and NC
        style_codes_ad = combined_data[:self.num_samples]
        style_codes_nc = combined_data[self.num_samples:]
        
        return style_codes_ad, style_codes_nc

    def plot_3d_tsne(self, perplexity=30):
        """
        Plot 3D TSNE using the AD and NC model latent spaces
        """
        # Generate style codes
        style_codes_ad, style_codes_nc = self.generate_style_codes(self.num_samples)
        
        # Combine data for scaling
        combined_data = np.vstack([style_codes_ad, style_codes_nc])
    
        scaler = StandardScaler()
        combined_data = scaler.fit_transform(combined_data)
        
        # Perform t-SNE with 3 components
        tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
        combined_tsne = tsne.fit_transform(combined_data)
        
        # Split TSNE results
        tsne_ad = combined_tsne[:self.num_samples]
        tsne_nc = combined_tsne[self.num_samples:]
        
        # Create enhanced 3D visualization
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(tsne_ad[:, 0], tsne_ad[:, 1], tsne_ad[:, 2], 
                   alpha=0.6, c='blue', label='AD', marker='o')
        ax.scatter(tsne_nc[:, 0], tsne_nc[:, 1], tsne_nc[:, 2], 
                   alpha=0.6, c='red', label='NC', marker='^')
        
        ax.set_title('3D t-SNE Visualization of StyleGAN Latent Space (W space)')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_zlabel('t-SNE Component 3')
        ax.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        plt.savefig("3D TSNE plot")
        
        style_codes_ad = combined_data[:self.num_samples]
        style_codes_nc = combined_data[self.num_samples:]
        
        return style_codes_ad, style_codes_nc



