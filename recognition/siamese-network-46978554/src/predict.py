"""
Example usage of trained model

usage: predict.py [-h] [-c CHECKPOINT] [-p] [-m MARGIN] images [images ...]

positional arguments:
  images                Image(s) to predict

options:
  -h, --help            show this help message and exit
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        Model checkpoint
  -p, --pretrained      Whether ResNet base is pretrained, default false
  -m MARGIN, --margin MARGIN
                        (TR/TS) margin for contrastive loss, default 0.2
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from dataset import MelanomaSkinCancerDataset
from modules import SiameseNetwork, init_classifier
from torchvision.io import read_image
from util import DATA_DIR


def main():
    torch.manual_seed(3710)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch version", torch.__version__, "on device", device)

    ref_set = MelanomaSkinCancerDataset(mode="ref")

    # Load model from checkpoint
    net = SiameseNetwork(pretrained=args.pretrained).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    net.load_state_dict(checkpoint)

    # Initialise a majority classifier from the reference dataset
    clf = init_classifier(net, ref_set, device, args.margin)

    # Load metadata so we can also report the ground truth labels
    metadata = pd.read_csv(DATA_DIR / "train-metadata.csv").iloc[:, 1:]

    with torch.no_grad():  # Disable gradient computation for efficiency
        for img_path in args.images:
            # Read image and normalise pixel values to [0, 1]
            img = read_image(img_path) / 255
            img = img.to(device)
            # Add extra first dimension (batch_size = 1)
            img = torch.unsqueeze(img, 0)

            # Forward pass through model to get embeddings
            embeddings = net(img).cpu()
            pred = clf.predict(embeddings)
            pred_proba = clf.predict_proba(embeddings)

            # Convert labels to names and compute the correct probability for printing
            isic_id = os.path.basename(img_path).split(".")[0]
            pred_name = "benign" if pred == 0 else "malignant"
            prob = round(float(1 - pred_proba if pred == 0 else pred_proba), 4)
            target = metadata["target"][metadata["isic_id"] == isic_id].iloc[0]
            target_name = "benign" if target == 0 else "malignant"

            print(f"{isic_id}:")
            print(f"  Prediction: {pred_name} ({prob})")
            print(f"  Target    : {target_name}")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "images", type=str, nargs="+", help="Image(s) to predict"
    )
    parser.add_argument("-c", "--checkpoint", type=str, help="Model checkpoint")
    parser.add_argument(
        "-p", "--pretrained", action="store_true", help="Whether ResNet base is pretrained, default false"
    )
    parser.add_argument(
        "-m", "--margin", type=float, default=0.2, help="(TR/TS) margin for contrastive loss, default 0.2"
    )

    # fmt: on

    args = parser.parse_args()

    model_path = Path(__file__).parent.parent / args.checkpoint

    main()
