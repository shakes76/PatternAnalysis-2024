from torchvision.transforms import ToPILImage
from stylegan2_pytorch import StyleGAN2


class Gan():

    def __init__(self,image_size, latent_dim, chanels, network_capacity) -> None:
        self.model = StyleGAN2(
            image_size=image_size,
            latent_dim=latent_dim,
            network_capacity= network_capacity
        )

        self.chanels = chanels
    
    def get_generator(self):
        return self.model.G

    def get_discriminator(self):
        return self.model.D
    
    def get_style_vector(self):
        return self.model.SE
    
    def initialise_weight(self):
        self.model._init_weights()

    