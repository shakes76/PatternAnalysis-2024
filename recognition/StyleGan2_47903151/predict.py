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

def generate_umap_plot(discriminator: Discriminator, loader, epoch, size=None):
    discriminator.eval()
    plt.figure(figsize=(10, 8))
    if size is None:
        # umap_embeddings = np.empty(loader.batch_size, loader)
        # labels =
        for i, (real, labels) in enumerate(loader):
            with torch.no_grad():
                embeddings = discriminator(real)
            reducer = umap.UMAP()
            umap_embeddings = reducer.fit_transform(embeddings)
            plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='viridis', s=5)
            plt.show()
    else:
        for i, (real, labels) in enumerate(loader):
            with torch.no_grad():
                embeddings = discriminator(real)
            reducer = umap.UMAP()
            umap_embeddings = reducer.fit_transform(embeddings)
            plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap='viridis', s=5)
            if i > size:
                break
    plt.colorbar()
    plt.title('UMAP Projection of Embeddings')
    print("Finished generating UMAP plot")
    plt.savefig(f"umap_epoch_{epoch}.png")


def load_model(path="model"):
    generator = Generator(LOG_RESOLUTION, W_DIM)
    discriminator = Discriminator(LOG_RESOLUTION)
    mapping_net = MappingNetwork(Z_DIM, W_DIM)
    plp = PathLengthPenalty(0.99)
    optim_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    optim_critic = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    optim_map = optim.Adam(mapping_net.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    if path is None:
        return generator, discriminator, mapping_net, plp, optim_gen, optim_critic, optim_map
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        try:
            if DEVICE == 'cpu':
                generator.load_state_dict(torch.load(f"{path}/generator.pth", map_location=DEVICE))
                discriminator.load_state_dict(torch.load(f"{path}/discriminator.pth", map_location=DEVICE))
                mapping_net.load_state_dict(torch.load(f"{path}/mapping.pth", map_location=DEVICE))
                plp.load_state_dict(torch.load(f"{path}/PLP.pth", map_location=DEVICE))
                a = torch.load(f"{path}/generator_opt.pth", map_location=DEVICE)
                optim_gen.load_state_dict(torch.load(f"{path}/generator_opt.pth", map_location=DEVICE))
                b = torch.load(f"{path}/discriminator_opt.pth", map_location=DEVICE)
                optim_critic.load_state_dict(torch.load(f"{path}/discriminator_opt.pth",
                                                        map_location=DEVICE,
                                                        weights_only=True))
                optim_map.load_state_dict(torch.load(f"{path}/mapping_opt.pth", map_location=DEVICE))
            else:
                generator.load_state_dict(torch.load(f"{path}/generator.pth"))
                discriminator.load_state_dict(torch.load(f"{path}/discriminator.pth"))
                mapping_net.load_state_dict(torch.load(f"{path}/mapping.pth"))
                plp.load_state_dict(torch.load(f"{path}/PLP.pth"))
                optim_gen.load_state_dict(torch.load(f"{path}/generator_opt.pth"))
                optim_critic.load_state_dict(torch.load(f"{path}/discriminator_opt.pth"))
                optim_map.load_state_dict(torch.load(f"{path}/mapping_opt.pth"))
        except Exception as err:
            print("Failed to load model. Training on new model instead.")
            # raise err

    return generator, discriminator, mapping_net, plp, optim_gen, optim_critic, optim_map


def generate_examples(gen, mapping_net, epoch, n=10, display=False):
    gen.eval()
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
                img = gen(w, noise)
                img = img[0]
                fig.add_subplot(rows, columns, i+1)
                plt.imshow(img.permute(1, 2, 0))
                plt.axis('off')
        plt.show()
    else:
        for i in range(n):
            with torch.no_grad():  # turn off gradient calculation to speed up generation
                w = get_w(1, W_DIM, DEVICE, mapping_net, LOG_RESOLUTION)
                noise = get_noise(1, LOG_RESOLUTION, DEVICE)
                img = gen(w, noise)
                if not os.path.exists(f'saved_examples/epoch{epoch}'):
                    os.makedirs(f'saved_examples/epoch{epoch}')
                save_image(img * 0.5 + 0.5, f"saved_examples/epoch{epoch}/img_{i}.png")
    gen.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Directory for the dataset")
    parser.add_argument("--model_dir", type=str, help="Directory of the saved model, if any")
    parser.add_argument("--load_model", type=bool, help="Choose whether to load model or not")
    parser.set_defaults(dataset_dir="AD_NC/train", model_dir="model", load_model=True)
    args = parser.parse_args()
    loader = get_loader(LOG_RESOLUTION, BATCH_SIZE)
    if args.load_model:
        gen, critic, mapping, plp, opt_gen, opt_critic, opt_mapping = load_model(args.model_dir)
    else:
        gen, critic, mapping, plp, opt_gen, opt_critic, opt_mapping = load_model(None)

    # with open("params/data.json", 'r') as f:
    #     json_data = json.load(f)
    # total_epochs = json_data["epochs"]
    # generator_loss = json_data["G_loss"]
    # discriminator_loss = json_data["D_loss"]
    total_epochs = 10
    generate_examples(gen, mapping, total_epochs, display=True)
    generate_umap_plot(critic, loader, total_epochs, 5)

