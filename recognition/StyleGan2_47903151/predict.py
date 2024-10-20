"""
Shows example usage of model.
"""
import os

from modules import *
from constants import *
import umap
from dataset import get_loader
from train import generate_examples, load_model
import argparse
import json



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Directory for the dataset")
    parser.add_argument("--model_dir", type=str, help="Directory of the saved model, if any")
    parser.add_argument("--load_model", type=bool, help="Choose whether to load model or not")
    parser.set_defaults(dataset_dir="AD_NC", model_dir="model", load_model=True)
    args = parser.parse_args()
    loader = get_loader(LOG_RESOLUTION, BATCH_SIZE)
    if args.load_model:
        gen, critic, mapping, plp, opt_gen, opt_critic, opt_mapping = load_model(args.model_dir)
    else:
        gen, critic, mapping, plp, opt_gen, opt_critic, opt_mapping = load_model(None)

    with open("params/data.json", 'r') as f:
        json_data = json.load(f)
    total_epochs = json_data["epochs"]
    generator_loss = json_data["G_loss"]
    discriminator_loss = json_data["D_loss"]

    generate_examples(gen, mapping, total_epochs, display=True)





