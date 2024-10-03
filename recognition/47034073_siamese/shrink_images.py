import pathlib
import time

import torch
from torch.utils.data import DataLoader
from torchvision.io import write_jpeg

from dataset import AllTumorDataset

DATA_DIR = pathlib.Path("data")
IMAGES_DIR = DATA_DIR / "train"
SMALL_IMAGES_DIR = DATA_DIR / "small_images"


def main() -> None:
    SMALL_IMAGES_DIR.mkdir(exist_ok=True)
    dataset = AllTumorDataset(IMAGES_DIR)
    loader = DataLoader(dataset, batch_size=1)

    start_time = time.time()
    n = 0
    for image, image_name in loader:
        n += 1
        image = (image[0] * 255).to(torch.uint8)
        write_jpeg(image, str(SMALL_IMAGES_DIR / f"{image_name[0]}.jpg"))

        if time.time() - start_time > 60:
            print(f"{n}/{len(loader.dataset)}")
            start_time = time.time()


if __name__ == "__main__":
    main()
