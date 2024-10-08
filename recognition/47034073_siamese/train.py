import time
import argparse
import logging
import pathlib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler
import matplotlib.pyplot as plt

from trainer import SiameseController, HyperParams
from dataset import TumorClassificationDataset

logger = logging.getLogger(__name__)

DATA_PATH = pathlib.Path(
    "C:/Users/Mitch/GitHub/PatternAnalysis-2024/recognition/47034073_siamese/data"
)
PAIRS_PATH = DATA_PATH / "pairs.csv"
ALL_META_PATH = DATA_PATH / "all.csv"
TRAIN_META_PATH = DATA_PATH / "train.csv"
VAL_META_PATH = DATA_PATH / "val.csv"
IMAGES_PATH = DATA_PATH / "small_images"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-l", "--load-model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    script_start_time = time.time()

    # Training params
    num_workers = 3
    learning_rate = 0.0001

    hparams = HyperParams(
        batch_size=128, num_epochs=2, learning_rate=learning_rate, weight_decay=0
    )
    if args.debug:
        hparams = HyperParams(batch_size=128, num_epochs=2, learning_rate=learning_rate)
    trainer = SiameseController(
        hparams, model_name="debug" if args.debug else "most_recent"
    )

    # Prepare data
    # all_df = pd.read_csv(ALL_META_PATH)
    # train_meta_df, val_meta_df = train_test_split(
    #     all_df, random_state=42, test_size=0.2
    # )
    train_meta_df = pd.read_csv(TRAIN_META_PATH)
    dataset = TumorClassificationDataset(IMAGES_PATH, train_meta_df)
    sampler = MPerClassSampler(
        labels=train_meta_df["target"],
        m=64,
        length_before_new_iter=1_000 if args.debug else 30_000,
    )
    train_loader = DataLoader(
        dataset,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
        sampler=sampler,
        batch_size=hparams.batch_size,
    )

    logger.info("Starting training...")

    if args.load_model is not None:
        trainer.load_model(args.load_model)
    else:
        trainer.train(train_loader)

    # Undersample to handle class imbalance
    # benign = train_meta_df[train_meta_df["target"] == 0]
    # malignant = train_meta_df[train_meta_df["target"] == 1]
    # num_malignant = len(malignant)
    # logger.debug("num malignant %d", num_malignant)
    # benign = benign.sample(random_state=42, n=num_malignant)
    # balanced_df = pd.concat([benign, malignant])
    # balanced_ds = TumorClassificationDataset(IMAGES_PATH, balanced_df)

    train_classification_loader = DataLoader(
        dataset,
        batch_size=hparams.batch_size,
        num_workers=num_workers,
    )

    logger.info("Fitting KNN...")
    embeddings, labels = trainer.compute_all_embeddings(train_classification_loader)
    knn = KNeighborsClassifier(
        n_neighbors=50 if args.debug else 26, weights="distance", p=2
    )
    embeddings = normalize(embeddings)

    fit_knn = knn.fit(embeddings, labels)
    logger.debug("KNN classes %s", fit_knn.classes_)

    logger.info("Evaluating classification on train data...")

    predictions = fit_knn.predict(embeddings)
    proba = fit_knn.predict_proba(embeddings)
    report = classification_report(labels, predictions)
    print(proba)
    print(labels)
    auc = roc_auc_score(labels, proba[:, 1])
    logger.info("train data report:\n%s\nauc: %f", report, auc)
    RocCurveDisplay.from_predictions(labels, proba[:, 1])
    plt.savefig("plots/train_roc")

    # Prepare validation data
    val_meta_df = pd.read_csv(VAL_META_PATH)

    val_dataset = TumorClassificationDataset(IMAGES_PATH, val_meta_df)
    val_loader = DataLoader(
        val_dataset, batch_size=hparams.batch_size, num_workers=num_workers
    )

    logger.info("Evaluating classification on val data...")

    embeddings, labels = trainer.compute_all_embeddings(val_loader)
    embeddings = normalize(embeddings)
    predictions = fit_knn.predict(embeddings)
    proba = fit_knn.predict_proba(embeddings)
    print(proba)

    report = classification_report(labels, predictions)
    auc = roc_auc_score(labels, proba[:, 1])
    logger.info("val data report\n%s\nauc: %f", report, auc)
    RocCurveDisplay.from_predictions(labels, proba[:, 1])
    plt.savefig("plots/val_roc")

    total_script_time = time.time() - script_start_time
    logger.info("Script done in %d seconds.", total_script_time)


if __name__ == "__main__":
    main()
