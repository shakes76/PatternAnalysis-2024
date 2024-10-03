import logging
import pathlib

import pandas as pd
from torch.utils.data import DataLoader

from modules import TumorClassifier, HyperParams
from dataset import TumorClassificationDataset, TumorPairDataset

logger = logging.getLogger(__name__)

DATA_PATH = pathlib.Path(
    "/home/Student/s4703407/PatternAnalysis-2024/recognition/47034073_siamese/data"
)
PAIRS_PATH = DATA_PATH / "pairs.csv"
TRAIN_META_PATH = DATA_PATH / "train.csv"
VAL_META_PATH = DATA_PATH / "val.csv"
IMAGES_PATH = DATA_PATH / "train"


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    hparams = HyperParams(batch_size=128, num_epochs=1)
    trainer = TumorClassifier(hparams)
    pairs_df = pd.read_csv(PAIRS_PATH)
    pairs_df = pairs_df.sample(random_state=42, n=128)
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

    logger.info("Computing centroids...")
    train_meta_df = pd.read_csv(TRAIN_META_PATH)
    train_meta_df = train_meta_df.sample(n=128)
    centroid_dataset = TumorClassificationDataset(IMAGES_PATH, train_meta_df)
    centroid_loader = DataLoader(centroid_dataset, batch_size=128, num_workers=2)
    trainer.compute_centroids(centroid_loader)

    logger.info("Evaluating classification on train data...")
    acc = trainer.evaluate(centroid_loader)
    logger.info("Train acc: %e", acc)
    logger.info("Script done.")

    logger.info("Evaluating classification on val data...")
    val_meta_df = pd.read_csv(VAL_META_PATH)
    val_meta_df = val_meta_df.sample(n=128)
    val_dataset = TumorClassificationDataset(IMAGES_PATH, val_meta_df)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=2)
    val_acc = trainer.evaluate(val_loader)
    logger.info("Val acc: %e", val_acc)


if __name__ == "__main__":
    main()
