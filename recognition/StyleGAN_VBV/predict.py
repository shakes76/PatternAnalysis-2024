import torch
from torchvision.utils import save_image
from modules import Generator
import os

# Constants
Z_DIM = 512
W_DIM = 512
IN_CHANNELS = 512
CHANNELS_IMG = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_examples(gen, steps, n=100):
    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, Z_DIM).to(DEVICE)
            img = gen(noise, alpha, steps)
            if not os.path.exists(f'saved_examples/step{steps}'):
                os.makedirs(f'saved_examples/step{steps}')
            save_image(img * 0.5 + 0.5, f"saved_examples/step{steps}/img_{i}.png")
    gen.train()