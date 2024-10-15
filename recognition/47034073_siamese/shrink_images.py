"""Shrinks all lesion images to be 224x224"""

import pathlib
import time

import torch
from torch.utils.data import DataLoader
from torchvision.io import write_jpeg

from dataset import ShrinkLesionDataset

DATA_DIR = pathlib.Path("data")
IMAGES_DIR = DATA_DIR / "train"
SMALL_IMAGES_DIR = DATA_DIR / "small_images"


def main() -> None:
    """Runs program"""
    SMALL_IMAGES_DIR.mkdir(exist_ok=True)
    dataset = ShrinkLesionDataset(IMAGES_DIR)
    loader = DataLoader(dataset, batch_size=512, num_workers=6)

    start_time = time.time()
    n = 0
    for images, image_names in loader:
        for i, image in enumerate(images):
            n += 1
            image_name = image_names[i]
            image = (image * 255).to(torch.uint8)
            write_jpeg(image, str(SMALL_IMAGES_DIR / f"{image_name}.jpg"))

            if time.time() - start_time > 30:
                print(f"{n}/{len(loader.dataset)}")
                start_time = time.time()


if __name__ == "__main__":
    main()
