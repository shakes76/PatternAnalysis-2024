import time
import argparse
import logging
import pathlib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from trainer import SiameseController, HyperParams
from dataset import TumorClassificationDataset

logger = logging.getLogger(__name__)

DATA_PATH = pathlib.Path("data")
ALL_META_PATH = DATA_PATH / "all.csv"
TRAIN_META_PATH = DATA_PATH / "train.csv"
VAL_META_PATH = DATA_PATH / "val.csv"
TEST_META_PATH = DATA_PATH / "test.csv"
IMAGES_PATH = DATA_PATH / "small_images"


def main() -> None:
    """Handle training, validation and testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-c", "--continue-training", action="store_true")
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("-l", "--load-model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    script_start_time = time.time()

    # Training params
    num_workers = 2
    learning_rate = 0.0001
    model_name = "most_recent"

    hparams = HyperParams(
        batch_size=128,
        num_epochs=1,
        learning_rate=learning_rate,
        margin=0.1,
    )
    if args.debug:
        hparams = HyperParams(batch_size=128, num_epochs=2, learning_rate=learning_rate)

    trainer = SiameseController(
        hparams, model_name="debug" if args.debug else model_name
    )

    # Training data
    train_meta_df = pd.read_csv(TRAIN_META_PATH)
    dataset = TumorClassificationDataset(IMAGES_PATH, train_meta_df, transform=True)

    sampler = MPerClassSampler(
        labels=train_meta_df["target"],
        m=hparams.batch_size / 2,
        length_before_new_iter=1_000 if args.debug else len(train_meta_df),
    )
    train_loader = DataLoader(
        dataset,
        pin_memory=True,
        num_workers=num_workers,
        # sampler=sampler,
        shuffle=True,
        batch_size=hparams.batch_size,
    )

    if args.continue_training and args.load_model is None:
        raise ValueError(
            "Cannot continue training without loading a model use -l option"
        )

    # Train or load embedding model
    if args.load_model is not None:
        logger.info("Loading model...")
        trainer.load_model(args.load_model)

    if args.load_model is None or args.continue_training:
        logger.info("Starting training...")
        trainer.train(train_loader)

    plt.figure()
    plt.plot(trainer.losses)
    plt.savefig("plots/train_loss")

    # Undersample to alleviate class imbalance
    benign = train_meta_df[train_meta_df["target"] == 0]
    malignant = train_meta_df[train_meta_df["target"] == 1]
    benign = benign.sample(random_state=42, n=len(malignant))
    knn_df = pd.concat([benign, malignant])

    knn_df = train_meta_df

    knn_ds = TumorClassificationDataset(IMAGES_PATH, knn_df, transform=False)

    # Undersample dataloader for KNN embeddings
    train_classification_loader = DataLoader(
        knn_ds,
        batch_size=hparams.batch_size,
        num_workers=num_workers,
    )
    embeddings, labels = trainer.compute_all_embeddings(train_classification_loader)
    logger.info("Embeddings \n%s", embeddings)

    embeddings = normalize(embeddings)

    # PCA
    logger.info("Fitting pca...")
    pca = PCA(n_components=2)
    pca_projections = pca.fit_transform(embeddings)
    logger.info("Plotting pca...")
    plt.figure()
    plt.scatter(
        pca_projections[:, 0],
        pca_projections[:, 1],
        c=knn_df["target"],
        cmap="coolwarm",
        marker=".",
        s=0.5,
    )
    plt.xlabel("component1")
    plt.ylabel("component2")
    benign_patch = mpatches.Patch(color="blue", label="Benign")
    malignant_patch = mpatches.Patch(color="red", label="Malignant")
    plt.legend(handles=[benign_patch, malignant_patch])
    logger.info("Writing image")
    plt.savefig("plots/train_pca")

    # tsne
    # logger.info("Fitting tsne...")
    # tsne = TSNE(random_state=42)
    # tsne_projections = tsne.fit_transform(embeddings)
    # logger.info("Plotting tsne...")
    # plt.figure()
    # plt.scatter(
    #     tsne_projections[:, 0],
    #     tsne_projections[:, 1],
    #     c=knn_df["target"],
    #     cmap="coolwarm",
    #     marker=".",
    # )
    # plt.xlabel("component1")
    # plt.ylabel("component2")
    # benign_patch = mpatches.Patch(color="blue", label="Benign")
    # malignant_patch = mpatches.Patch(color="red", label="Malignant")
    # plt.legend(handles=[benign_patch, malignant_patch])
    # logger.info("Writing image")
    # plt.savefig("plots/train_tsne")

    # Fit KNN
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", p=2)
    logger.info("Fitting KNN...")
    fit_knn = knn.fit(embeddings, labels)

    # Eval on train
    logger.info("Evaluating classification on train data...")
    _evaluate_classification(fit_knn, embeddings, labels, data_name="train")

    # Validation data
    val_meta_df = pd.read_csv(VAL_META_PATH)
    val_dataset = TumorClassificationDataset(IMAGES_PATH, val_meta_df, transform=False)
    val_loader = DataLoader(
        val_dataset, batch_size=hparams.batch_size, num_workers=num_workers
    )
    embeddings, labels = trainer.compute_all_embeddings(val_loader)
    embeddings = normalize(embeddings)

    # Eval on validation
    logger.info("Evaluating classification on val data...")
    _evaluate_classification(fit_knn, embeddings, labels, data_name="val")

    if args.test:
        test_meta_df = pd.read_csv(TEST_META_PATH)
        test_dataset = TumorClassificationDataset(
            IMAGES_PATH, test_meta_df, transform=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=hparams.batch_size, num_workers=num_workers
        )
        embeddings, labels = trainer.compute_all_embeddings(test_loader)
        embeddings = normalize(embeddings)

        logger.info("Evaluating classification on test data...")
        _evaluate_classification(fit_knn, embeddings, labels, data_name="test")

    total_script_time = time.time() - script_start_time
    logger.info("Script done in %d seconds.", total_script_time)


def _evaluate_classification(
    knn: KNeighborsClassifier,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    data_name: str = "train",
) -> None:
    predictions = knn.predict(embeddings)
    proba = knn.predict_proba(embeddings)
    report = classification_report(labels, predictions)
    auc = roc_auc_score(labels, proba[:, 1])
    logger.info("%s data report:\n%s\nauc: %f", data_name, report, auc)
    RocCurveDisplay.from_predictions(labels, proba[:, 1])
    plt.savefig(f"plots/{data_name}_roc")


if __name__ == "__main__":
    main()
