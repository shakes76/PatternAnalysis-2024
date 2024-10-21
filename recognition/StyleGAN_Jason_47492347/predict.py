from utils import *
from settings import *
from modules import Generator
import argparse

def generate_images(model_path, output_path, num_images, seed=0):
    """
    Generate images using a pre-trained StyleGAN model.

    Parameters:
    - model_path (str): Path to the pre-trained model.
    - output_path (str): Directory where the generated images will be saved.
    - num_images (int): Number of images to generate.
    - seed (int): Seed for random number generator (default: 0).
    """
    # load model
    gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
    gen.load_state_dict(torch.load(model_path, weights_only=True))
    gen.eval()

    # set seed
    torch.manual_seed(seed)

    # set step and alpha for full size fully trained model
    step = int(log2(IMG_SIZE / 4))
    alpha = 1.0

    # generate images
    for i in range(num_images):
        with torch.no_grad():
            noise = torch.randn(1, Z_DIM).to(DEVICE)  # Random noise as (seeded) input
            img = gen(noise, alpha, steps=step)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            vutils.save_image(img*0.5+0.5, f"{output_path}/img_{i}.png")

    gen.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using a pre-trained StyleGAN model.")
    parser.add_argument("--model", type=str, required=True, help="Name of the pre-trained model (.pt file)")
    parser.add_argument("--output", type=str, required=True, help="Desired name for the output directory")
    parser.add_argument("--n", type=int, required=True, help="Number of images to generate")
    parser.add_argument("--seed", type=int, default=0, help="Set RNG seed for reproducibility")
    args = parser.parse_args()

    model_path = f"{SRC}/saved_models/{args.model}.pt"
    output_path = f"{SRC}/predict_outputs/{args.output}"
    generate_images(model_path, output_path, args.n, args.seed)
