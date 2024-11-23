"""
Shows example usage of model.
"""
import os

import torch
import matplotlib.image as mpimg

from modules import *
from constants import *
import umap
from dataset import get_loader
import argparse
import json
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def umap_plot(mapping_AD: MappingNetwork, mapping_NC: MappingNetwork):
    """
    Plot and shows the umap plot of the style embeddings, using the mapping network of model trained on AD and NC
    classes.
    """
    mapping_AD.eval()
    mapping_NC.eval()
    plt.figure(figsize=(10, 8))

    with torch.no_grad():

        z = torch.randn(1, W_DIM).to(DEVICE)
        style_AD = mapping_AD(z)
        style_NC = mapping_NC(z)
        for i in range(10000):
            z = torch.randn(1, W_DIM).to(DEVICE)
            style_AD = torch.cat((mapping_AD(z), style_AD))
            style_NC = torch.cat((mapping_NC(z), style_NC))
        reducer = umap.UMAP()
        umap_embeddings = reducer.fit_transform(style_AD)
        plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=5)
        umap_embeddings = reducer.fit_transform(style_NC)
        plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=5)
        plt.title("Umap plot of embeddings, epoch 5")
        plt.legend(["AD", "NC"])
    plt.show()


def load_model(path=""):
    """
    load the model at the given directory. If the directory is None, use new model.
    :param: path: path of the saved/pretrained model.
    :return: generator, discriminator, mapping network and path length penalty
    """
    generator = Generator(LOG_RESOLUTION, W_DIM)
    discriminator = Discriminator(LOG_RESOLUTION)
    mapping_net = MappingNetwork(Z_DIM, W_DIM)
    plp = PathLengthPenalty(0.99)
    if path is None:
        print("Using untrained model.")
        return generator, discriminator, mapping_net, plp
    if not os.path.exists(path):
        print("Model path does not exits. An untrained model will be used.")
        os.mkdir(path)
    else:
        try:
            if DEVICE == 'cpu':
                generator.load_state_dict(torch.load(f"{path}/generator.pth",
                                                     map_location=DEVICE,
                                                     weights_only=True))
                mapping_net.load_state_dict(torch.load(f"{path}/mapping.pth",
                                                       map_location=DEVICE,
                                                       weights_only=True))
                discriminator.load_state_dict(torch.load(f"{path}/discriminator.pth",
                                                         map_location=DEVICE,
                                                         weights_only=True))
                plp.load_state_dict(torch.load(f"{path}/PLP.pth",
                                               map_location=DEVICE,
                                               weights_only=True))
            else:
                generator.load_state_dict(torch.load(f"{path}/generator.pth", weights_only=True))
                mapping_net.load_state_dict(torch.load(f"{path}/mapping.pth", weights_only=True))
                discriminator.load_state_dict(torch.load(f"{path}/discriminator.pth", weights_only=True))
                plp.load_state_dict(torch.load(f"{path}/PLP.pth", weights_only=True))
        except Exception as err:
            print("Failed to load model. Training on new model instead.")

    return generator, discriminator, mapping_net, plp

def load_optimizers(generator: Generator, discriminator: Discriminator, mapping_net: MappingNetwork, path="model"):
    """
    Loads optimizers from the given directory
    :return: generator optimizer, discriminator optimizer, and mapping network optimizer
    """
    optim_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    optim_critic = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    optim_map = optim.Adam(mapping_net.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
    if path is None:
        return optim_gen, optim_critic, optim_map
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(f"{path}/generator_opt.pth"):
        return optim_gen, optim_critic, optim_map
    else:
        if DEVICE == 'cpu':
            optim_gen.load_state_dict(torch.load(f"{path}/generator_opt.pth", map_location=DEVICE, weights_only=True))
            optim_critic.load_state_dict(torch.load(f"{path}/discriminator_opt.pth",
                                                    map_location=DEVICE,
                                                    weights_only=True))
            optim_map.load_state_dict(torch.load(f"{path}/mapping_opt.pth", map_location=DEVICE, weights_only=True))
        else:
            optim_gen.load_state_dict(torch.load(f"{path}/generator_opt.pth", weights_only=True))
            optim_critic.load_state_dict(torch.load(f"{path}/discriminator_opt.pth", weights_only=True))
            optim_map.load_state_dict(torch.load(f"{path}/mapping_opt.pth", weights_only=True))
        return optim_gen, optim_critic, optim_map


def generate_examples(generator: Generator, mapping_net: MappingNetwork, epoch=None, n=10, display=False, model_dir=""):
    """
    Use generator and mapping network to generate a few images, and save the generated images.
    :param epoch the number of epoch the model have trained for
    :param n: integer, the number of example wish to generate
    :param display: Choose whether to display the image or not.
    :returns a list of generated images.
    """
    generator.eval()
    images = []
    for i in range(n):
        with torch.no_grad():  # turn off gradient calculation to speed up generation
            w = get_w(1, W_DIM, DEVICE, mapping_net, LOG_RESOLUTION)
            noise = get_noise(1, LOG_RESOLUTION, DEVICE)
            img = generator(w, noise)
            if epoch is None:
                if not os.path.exists(f'{model_dir}/saved_examples'):
                    os.makedirs(f'{model_dir}/saved_examples')
                save_image(img*0.5+0.5, f"{model_dir}/saved_examples/img_{i}.png")
                images.append(np.asarray(mpimg.imread(f"{model_dir}/saved_examples/img_{i}.png")))
            else:
                if not os.path.exists(f'{model_dir}/saved_examples/epoch{epoch}'):
                    os.makedirs(f'{model_dir}/saved_examples/epoch{epoch}')
                save_image(img*0.5+0.5, f"{model_dir}/saved_examples/epoch{epoch}/img_{i}.png")
                images.append(np.asarray(mpimg.imread(f"{model_dir}/saved_examples/img_{i}.png")))
    generator.train()
    return images




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Directory for the dataset")
    parser.add_argument("--model_dir", type=str, help="Directory of the saved model, if any")
    parser.add_argument("--load_model", type=bool, help="Choose whether to load model or not")
    parser.add_argument("--plot_umap", type=bool, help="Plot umap or not")
    parser.add_argument("--AD_dir", type=str, help="directory of model trained on AD, if any")
    parser.add_argument("--NC_dir", type=str, help="directory of model trained on NC, if any")
    parser.set_defaults(dataset_dir="AD_NC/train", model_dir="model_epoch_30", load_model=True, plot_umap=False)
    args = parser.parse_args()
    loader = get_loader(LOG_RESOLUTION, BATCH_SIZE)
    if args.load_model:
        gen, critic, mapping, plp = load_model(args.model_dir)
        # opt_gen, opt_critic, opt_mapping = load_optimizers(gen, critic, mapping, args.model_dir)
    else:
        gen, critic, mapping, plp = load_model(None)
    imgs = generate_examples(gen, mapping, display=True, model_dir=args.model_dir)
    real, _ = next(iter(loader))  # get a batch
    fig, axes = plt.subplots(2, min([real.size(0), len(imgs), 5]), figsize=(10, 5))
    fig.suptitle("Real images vs. Fake images, fakes in row 0, reals in row 1")
    for i in range(min([real.size(0), len(imgs), 5])):
        axes[0, i].imshow(imgs[i])
        axes[0, i].axis("off")
        axes[1, i].imshow((real[i]*0.5+0.5).permute(1, 2, 0))
        axes[1, i].axis("off")
    plt.show()


    if args.plot_umap:
        gen_AD, _, map_AD, _ = load_model(args.AD_dir)
        gen_NC, _, map_NC, _ = load_model(args.NC_dir)
        umap_plot(map_AD, map_NC)
