from utils import *
from settings import *
from modules import Generator

def generate_images(gen_path, label, num_images, seed=0):
    """
    Generate images using a pre-trained StyleGAN model.
    """
    gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
    gen.load_state_dict(torch.load(gen_path))
    torch.manual_seed(seed)
    step = int(log2(IMG_SIZE / 4))

    generate_examples(gen, step, num_images, label)

if __name__ == "__main__":
    path = f"{SRC}/saved_models/gen_{MODEL_LABEL}.pt"
    generate_images(path, GENERATE_LABEL, 10)
