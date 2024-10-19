from PIL import Image
import torch
from torchvision.transforms import ToPILImage
from stylegan2_pytorch import StyleGAN2

# Set up the parameters
latent_dim = 512
image_size = 256
num_channels = 3

# Initialize the StyleGAN2 model
model = StyleGAN2(
    image_size=image_size,
    latent_dim=latent_dim,
    network_capacity=16
)

model.eval()

z = torch.randn(1, latent_dim)


styles = model.mapping_network(z)


fake_image = model.synthesis_network(styles)

# Post-process the image tensor (assuming the output is in range [-1, 1])
# Convert it to the range [0, 1]
fake_image = (fake_image + 1) * 0.5

# Convert the generated image tensor to a PIL image for saving
to_pil = ToPILImage()
image_pil = to_pil(fake_image.squeeze(0).cpu().detach())

# Save the image
image_pil.save("fake_face_image.png")

print("Image saved as fake_face_image.png")
