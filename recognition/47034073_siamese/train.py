import logging
import pathlib

import pandas as pd
from torch.utils.data import DataLoader

from modules import TumorTrainer, HyperParams
from dataset import TumorPairDataset

logger = logging.getLogger(__name__)

DATA_PATH = pathlib.Path(
    "/home/Student/s4703407/PatternAnalysis-2024/recognition/47034073_siamese/data"
)
PAIRS_PATH = DATA_PATH / "pairs.csv"
IMAGES_PATH = DATA_PATH / "train"


def main():
    logging.basicConfig(level=logging.DEBUG)
    hparams = HyperParams()
    trainer = TumorTrainer(hparams)
    pairs_df = pd.read_csv(PAIRS_PATH)
    dataset = TumorPairDataset(IMAGES_PATH, pairs_df)
    train_loader = DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=hparams.batch_size,
        num_workers=2,
        drop_last=True,
    )
    logger.info("Starting training...")
    trainer.train(train_loader)


if __name__ == "__main__":
    main()
