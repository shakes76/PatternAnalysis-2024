"""
Shows example usage of model.
"""
import os

from modules import *
from constants import *
import umap
from dataset import get_loader
import argparse
import json
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# def generate_umap_plot(discriminator: Discriminator, loader, epoch, size=None):
#     discriminator.eval()
#     plt.figure(figsize=(10, 8))
#     if size is None:
#         # umap_embeddings = np.empty(loader.batch_size, loader)
#         # labels =
#         for i, (real, labels) in enumerate(loader):
#             with torch.no_grad():
#                 embeddings = discriminator(real)
#             reducer = umap.UMAP()
#             umap_embeddings = reducer.fit_transform(embeddings)
#             plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='viridis', s=5)
#             plt.show()
#     else:
#         for i, (real, labels) in enumerate(loader):
#             with torch.no_grad():
#                 embeddings = discriminator(real)
#             reducer = umap.UMAP()
#             umap_embeddings = reducer.fit_transform(embeddings)
#             plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='viridis', s=5)
#             if i > size:
#                 break
#     plt.colorbar()
#     plt.title('UMAP Projection of Embeddings')
#     print("Finished generating UMAP plot")
#     plt.savefig(f"umap_epoch_{epoch}.png")

def umap_plot(generator_ad: Generator, generator_nc: Generator, mapping_net: MappingNetwork, path: str):
    """
    :param generator_ad: The generator trained on the AD style
    :param generator_nc: Generator trained on NC style
    :param mapping_net: Mapping network that generates noise
    :param path: path to save the image to
    :return:
    """
    pass


def load_model(path=""):
    """
    :param: path: load the model at the given directory.
    If the directory is None, use new model.
    :return: generator, discriminator, noise mapping network and path length penalty
    """
    print(path)
    generator = Generator(LOG_RESOLUTION, W_DIM)
    discriminator = Discriminator(LOG_RESOLUTION)
    mapping_net = MappingNetwork(Z_DIM, W_DIM)
    plp = PathLengthPenalty(0.99)
    if path is None:
        return generator, discriminator, mapping_net, plp
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        try:
            if DEVICE == 'cpu':
                generator.load_state_dict(torch.load(f"{path}/generator.pth", map_location=DEVICE))
                discriminator.load_state_dict(torch.load(f"{path}/discriminator.pth", map_location=DEVICE))
                mapping_net.load_state_dict(torch.load(f"{path}/mapping.pth", map_location=DEVICE))
                plp.load_state_dict(torch.load(f"{path}/PLP.pth", map_location=DEVICE))
            else:
                generator.load_state_dict(torch.load(f"{path}/generator.pth"))
                discriminator.load_state_dict(torch.load(f"{path}/discriminator.pth"))
                mapping_net.load_state_dict(torch.load(f"{path}/mapping.pth"))
                plp.load_state_dict(torch.load(f"{path}/PLP.pth"))
        except Exception:
            print("Failed to load model. Training on new model instead.")

    return generator, discriminator, mapping_net, plp

def load_optimizers(generator: Generator, discriminator: Discriminator, mapping_net: MappingNetwork, path="model"):
    """
    Loads each optimizer from the given directory
    :return: generator optimizer, discriminator optimizer, and noise mapping network optimizer
    """
    optim_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    optim_critic = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    optim_map = optim.Adam(mapping_net.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    if path is None:
        return optim_gen, optim_critic, optim_map
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        if DEVICE == 'cpu':
            optim_gen.load_state_dict(torch.load(f"{path}/generator_opt.pth", map_location=DEVICE))
            optim_critic.load_state_dict(torch.load(f"{path}/discriminator_opt.pth",
                                                    map_location=DEVICE,
                                                    weights_only=True))
            optim_map.load_state_dict(torch.load(f"{path}/mapping_opt.pth", map_location=DEVICE))
        else:
            optim_gen.load_state_dict(torch.load(f"{path}/generator_opt.pth"))
            optim_critic.load_state_dict(torch.load(f"{path}/discriminator_opt.pth"))
            optim_map.load_state_dict(torch.load(f"{path}/mapping_opt.pth"))
        return optim_gen, optim_critic, optim_map


def generate_examples(generator: Generator, mapping_net: MappingNetwork, epoch, n=10, display=False):
    """
    Use generator and noise mapping network to generate a few images, then save the generated images.
    :param epoch the number of epoch the model have trained for
    :param n: integer, the number of example wish to generate
    :param display: Choose whether to display the image or not.
    """
    generator.eval()
    # alpha = 1.0
    if display:
        fig = plt.figure(figsize=(10, 7))
        plt.title(f"Epoch {epoch}")
        plt.axis('off')
        rows, columns = 2, 2
        for i in range(4):
            with torch.no_grad():
                w = get_w(1, W_DIM, DEVICE, mapping_net, LOG_RESOLUTION)
                noise = get_noise(1, LOG_RESOLUTION, DEVICE)
                img = generator(w, noise)
                img = img[0]
                plt.imshow(img.permute(1, 2, 0))
                plt.axis('off')
                plt.show()
    else:
        for i in range(n):
            with torch.no_grad():  # turn off gradient calculation to speed up generation
                w = get_w(1, W_DIM, DEVICE, mapping_net, LOG_RESOLUTION)
                noise = get_noise(1, LOG_RESOLUTION, DEVICE)
                img = generator(w, noise)
                if not os.path.exists(f'saved_examples/epoch{epoch}'):
                    os.makedirs(f'saved_examples/epoch{epoch}')
                save_image(img * 0.5 + 0.5, f"saved_examples/epoch{epoch}/img_{i}.png")
    generator.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Directory for the dataset")
    parser.add_argument("--model_dir", type=str, help="Directory of the saved model, if any")
    parser.add_argument("--load_model", type=bool, help="Choose whether to load model or not")
    parser.set_defaults(dataset_dir="AD_NC/train", model_dir="model_epoch_30", load_model=True)
    args = parser.parse_args()
    loader = get_loader(LOG_RESOLUTION, BATCH_SIZE)
    if args.load_model:
        gen, critic, mapping, plp = load_model(args.model_dir)
        # opt_gen, opt_critic, opt_mapping = load_optimizers(gen, critic, mapping, args.model_dir)
    else:
        gen, critic, mapping, plp = load_model(None)
        # opt_gen, opt_critic, opt_mapping = load_optimizers(gen, critic, mapping, None)
    total_epochs = 30
    generate_examples(gen, mapping, total_epochs, display=False)
    # generate_umap_plot(critic, loader, total_epochs, 5)

