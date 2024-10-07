import time
import argparse
import logging
import pathlib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler

from modules import SiameseController, HyperParams
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

    script_start_time = time.time()

    # Training params
    num_workers = 3

    hparams = HyperParams(batch_size=128, num_epochs=5, learning_rate=0.0001)
    if args.debug:
        hparams = HyperParams(batch_size=128, num_epochs=1)

    trainer = SiameseController(hparams)

    train_meta_df = pd.read_csv(TRAIN_META_PATH)
    dataset = TumorClassificationDataset(IMAGES_PATH, train_meta_df)

    # Prepare train pair data
    # pairs_df = pd.read_csv(PAIRS_PATH)
    # if args.debug:
    #     pairs_df = pairs_df.sample(random_state=42, n=128)
    # dataset = TumorPairDataset(IMAGES_PATH, pairs_df)

    sampler = MPerClassSampler(
        labels=train_meta_df["target"], m=64, batch_size=hparams.batch_size
    )
    train_loader = DataLoader(
        dataset,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        sampler=sampler,
    )

    logger.info("Starting training...")
    trainer.train(train_loader)

    # Undersample to handle class imbalance
    benign = train_meta_df[train_meta_df["target"] == 0]
    malignant = train_meta_df[train_meta_df["target"] == 1]
    num_malignant = len(malignant)
    logger.debug("num malignant %d", num_malignant)
    benign = benign.sample(random_state=42, n=num_malignant)
    train_meta_df = pd.concat([benign, malignant])

    train_classification_dataset = TumorClassificationDataset(
        IMAGES_PATH, train_meta_df
    )
    train_classification_loader = DataLoader(
        train_classification_dataset,
        batch_size=hparams.batch_size,
        num_workers=num_workers,
    )

    logger.info("Fitting KNN...")
    embeddings, labels = trainer.compute_all_embeddings(train_classification_loader)
    knn = KNeighborsClassifier(
        n_neighbors=2 if args.debug else 5, weights="distance", p=2
    )
    scaler = StandardScaler()
    scaler = scaler.fit(embeddings)
    embeddings = scaler.transform(embeddings)

    fit_knn = knn.fit(embeddings, labels)

    logger.info("Evaluating classification on train data...")

    predictions = fit_knn.predict(embeddings)
    proba = fit_knn.predict_proba(embeddings)
    report = classification_report(labels, predictions)
    auc = roc_auc_score(labels, proba[:, 1])
    logger.info("train data report:\n%s\nauc: %d", report, auc)

    # Prepare validation data
    val_meta_df = pd.read_csv(VAL_META_PATH)
    # if args.debug:
    #     val_meta_df = val_meta_df.sample(n=256)

    val_dataset = TumorClassificationDataset(IMAGES_PATH, val_meta_df)
    val_loader = DataLoader(
        val_dataset, batch_size=hparams.batch_size, num_workers=num_workers
    )

    logger.info("Evaluating classification on val data...")

    embeddings, labels = trainer.compute_all_embeddings(val_loader)
    embeddings = scaler.transform(embeddings)
    predictions = fit_knn.predict(embeddings)
    proba = fit_knn.predict_proba(embeddings)

    report = classification_report(labels, predictions)
    auc = roc_auc_score(labels, proba[:, 1])
    logger.info("val data report\n%s\nauc: %d", report, auc)

    total_script_time = time.time() - script_start_time
    logger.info("Script done in %d seconds.", total_script_time)


if __name__ == "__main__":
    main()
