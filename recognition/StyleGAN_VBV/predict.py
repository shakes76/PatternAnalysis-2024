import torch
from torchvision.utils import save_image
from modules import Generator

# Load your trained generator model
def load_model(checkpoint_path):
    model = Generator(Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def generate_and_save_images(model, n=100):
    for i in range(n):
        noise = torch.randn(1, Z_DIM).to(DEVICE)
        img = model(noise, alpha=1.0, steps=6)  # Adjust 'steps' based on your training
        save_image(img * 0.5 + 0.5, f"generated_images/img_{i}.png")

if __name__ == "__main__":
    model = load_model('path_to_your_checkpoint.pth')
    generate_and_save_images(model)
