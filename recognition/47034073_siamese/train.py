import time
import argparse
import logging
import pathlib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
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
IMAGES_PATH = DATA_PATH / "small_images"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    if args.debug:
        _debug()
        return

    _train()


def _debug() -> None:
    script_start_time = time.time()
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

    train_classification_dataset = TumorClassificationDataset(
        IMAGES_PATH, train_meta_df
    )
    train_classification_loader = DataLoader(
        train_classification_dataset, batch_size=128, num_workers=2
    )
    # trainer.compute_centroids(train_classification_loader)
    embeddings, labels = trainer.compute_all_embeddings(train_classification_loader)
    knn = KNeighborsClassifier()
    fit_knn = knn.fit(embeddings, labels)

    logger.debug(
        "embeddings shape %s\nlabels shape %s", str(embeddings.shape), str(labels.shape)
    )

    predictions = fit_knn.predict(embeddings)
    report = classification_report(labels, predictions)
    logger.info("train data report:\n%s", report)

    logger.info("Evaluating classification on train data...")
    # acc = trainer.evaluate(train_classification_loader)
    # logger.info("Train acc: %e", acc)

    logger.info("Evaluating classification on val data...")
    val_meta_df = pd.read_csv(VAL_META_PATH)
    val_meta_df = val_meta_df.sample(n=128)
    val_dataset = TumorClassificationDataset(IMAGES_PATH, val_meta_df)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=2)
    # val_acc = trainer.evaluate(val_loader)
    # logger.info("Val acc: %e", val_acc)
    embeddings, labels = trainer.compute_all_embeddings(val_loader)
    predictions = fit_knn.predict(embeddings)

    report = classification_report(labels, predictions)
    logger.info("val data report\n%s", report)

    total_script_time = time.time() - script_start_time
    logger.info("Script done in %d seconds.", total_script_time)


def _train() -> None:
    # Training params
    num_workers = 3
    hparams = HyperParams(batch_size=128, num_epochs=100, learning_rate=0.0001)
    trainer = TumorClassifier(hparams)

    # Prepare train pair data
    pairs_df = pd.read_csv(PAIRS_PATH)
    dataset = TumorPairDataset(IMAGES_PATH, pairs_df)
    train_loader = DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=hparams.batch_size,
        num_workers=num_workers,
        drop_last=True,
    )

    logger.info("Starting training...")
    trainer.train(train_loader)

    # Prepare train classification data
    train_meta_df = pd.read_csv(TRAIN_META_PATH)
    train_classification_dataset = TumorClassificationDataset(
        IMAGES_PATH, train_meta_df
    )
    train_classification_loader = DataLoader(
        train_classification_dataset,
        batch_size=hparams.batch_size,
        num_workers=num_workers,
    )

    logger.info("Computing centroids...")
    trainer.compute_centroids(train_classification_loader)

    logger.info("Evaluating classification on train data...")
    acc = trainer.evaluate(train_classification_loader)
    logger.info("Train acc: %e", acc)

    # Prepare validation data
    val_meta_df = pd.read_csv(VAL_META_PATH)
    val_dataset = TumorClassificationDataset(IMAGES_PATH, val_meta_df)
    val_loader = DataLoader(
        val_dataset, batch_size=hparams.batch_size, num_workers=num_workers
    )

    logger.info("Evaluating classification on val data...")
    val_acc = trainer.evaluate(val_loader)
    logger.info("Val acc: %e", val_acc)

    logger.info("Script done.")


if __name__ == "__main__":
    main()
