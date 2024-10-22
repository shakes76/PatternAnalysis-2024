import torch
import torch.nn as nn

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(z_dim, w_dim),
            nn.ReLU(),
            nn.Linear(w_dim, w_dim)
        )
    
    def forward(self, z):
        return self.mapping(z)

class StyleGANGenerator(nn.Module):
    def __init__(self, z_dim, w_dim, img_channels):
        super().__init__()
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.synthesis = nn.Sequential(
            nn.ConvTranspose2d(w_dim, 256, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        w = self.mapping(z)
        w = w.view(w.size(0), -1, 1, 1)
        generated_image = self.synthesis(w)
        return generated_image
    

if __name__ == "__main__":
    z_dim = 128  
    w_dim = 512  
    img_channels = 3  

    generator = StyleGANGenerator(z_dim=z_dim, w_dim=w_dim, img_channels=img_channels)

    generator.eval()

    z = torch.randn(1, z_dim) 

    with torch.no_grad():
        generated_image = generator(z)
        
    print(f"Output shape: {generated_image.shape}")