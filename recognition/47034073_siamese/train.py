"""Handle training, validation and testing of triamese network."""

import time
import argparse
import logging
import pathlib

from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.svm import SVC
from sklearn import neural_network
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import utils
from trainer import SiameseController, HyperParams
from dataset import LesionClassificationDataset

logger = logging.getLogger(__name__)

PLOTS_PATH = pathlib.Path("plots")
DATA_PATH = pathlib.Path("data")
ALL_META_PATH = DATA_PATH / "all.csv"
TRAIN_META_PATH = DATA_PATH / "train.csv"
VAL_META_PATH = DATA_PATH / "val.csv"
TEST_META_PATH = DATA_PATH / "test.csv"
IMAGES_PATH = DATA_PATH / "small_images"


def main() -> None:
    """Run the program."""
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
    learning_rate = 0.00001
    model_name = "most_recent"

    hparams = HyperParams(
        batch_size=128,
        num_epochs=1000,
        learning_rate=learning_rate,
        margin=1,
    )
    if args.debug:
        hparams = HyperParams(batch_size=128, num_epochs=2, learning_rate=learning_rate)

    trainer = SiameseController(
        hparams, model_name="debug" if args.debug else model_name
    )

    # Define classifiers
    knn = KNeighborsClassifier(
        n_neighbors=100, weights=lambda arr: _margin_weight(arr, hparams), p=2
    )
    svm = SVC(probability=True, random_state=42)
    nn = neural_network.MLPClassifier(
        hidden_layer_sizes=(64, 64, 64, 64, 64, 64, 64),
        learning_rate_init=0.0001,
        random_state=42,
        early_stopping=True,
        verbose=False,
    )

    # Siamese training data
    train_meta_df = pd.read_csv(TRAIN_META_PATH)
    dataset = LesionClassificationDataset(IMAGES_PATH, train_meta_df, transform=True)
    train_loader = DataLoader(
        dataset,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=True,
        batch_size=hparams.batch_size,
        drop_last=True,
    )

    # Classifier training data
    # Undersample to alleviate class imbalance
    benign = train_meta_df[train_meta_df["target"] == 0]
    malignant = train_meta_df[train_meta_df["target"] == 1]
    benign = benign.sample(random_state=42, n=len(malignant))
    knn_df = pd.concat([benign, malignant])
    knn_ds = LesionClassificationDataset(IMAGES_PATH, knn_df, transform=False)
    train_classification_loader = DataLoader(
        knn_ds,
        batch_size=hparams.batch_size,
        num_workers=num_workers,
    )

    # Validation data
    val_meta_df = pd.read_csv(VAL_META_PATH)
    val_dataset = LesionClassificationDataset(IMAGES_PATH, val_meta_df, transform=False)
    val_loader = DataLoader(
        val_dataset, batch_size=hparams.batch_size, num_workers=num_workers
    )

    knn_best_auc = 0
    svm_best_auc = 0
    nn_best_auc = 0

    def validate():
        """To run at the end of epoch to check performance of each classifier on validation data."""
        nonlocal knn_best_auc
        nonlocal svm_best_auc
        nonlocal nn_best_auc

        # Get train and validation embeddings
        embeddings, labels = trainer.compute_all_embeddings(train_classification_loader)
        embeddings = normalize(embeddings)
        val_embeddings, val_labels = trainer.compute_all_embeddings(val_loader)
        val_embeddings = normalize(val_embeddings)

        # Fit classifiers
        fit_nn = nn.fit(embeddings, labels)
        fit_knn = knn.fit(embeddings, labels)
        fit_svm = svm.fit(embeddings, labels)

        # Compute AUC scores
        knn_auc = _evaluate_classification(
            fit_knn, val_embeddings, val_labels, minimal=True
        )
        svm_auc = _evaluate_classification(
            fit_svm, val_embeddings, val_labels, minimal=True
        )
        nn_auc = _evaluate_classification(
            fit_nn, val_embeddings, val_labels, minimal=True
        )
        logger.info("\nsvm auc %e\nknn auc %e\nMLP auc %e", svm_auc, knn_auc, nn_auc)

        # Check for best auc so far
        if knn_auc > knn_best_auc:
            knn_best_auc = knn_auc
            trainer.save_model(f"{model_name}_best_knn")
        if svm_auc > svm_best_auc:
            svm_best_auc = svm_auc
            trainer.save_model(f"{model_name}_best_svm")
        if nn_auc > nn_best_auc:
            nn_best_auc = nn_auc
            trainer.save_model(f"{model_name}_best_nn")

    trainer.end_of_epoch_func = validate

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

    utils.plot_training_data(trainer)

    # Get classifer training observation embeddings
    embeddings, labels = trainer.compute_all_embeddings(train_classification_loader)
    logger.info("Embeddings \n%s", embeddings)
    embeddings = normalize(embeddings)

    utils.plot_pca(embeddings, knn_df["target"])
    utils.plot_tsne(embeddings, knn_df["target"])

    # Fit classifiers
    logger.info("Fitting KNN...")
    fit_knn = knn.fit(embeddings, labels)
    logger.info("Fitting SVM")
    fit_svm = svm.fit(embeddings, labels)
    logger.info("Fitting NN...")
    fit_nn = nn.fit(embeddings, labels)

    # Eval on train
    logger.info("Evaluating classification on train data...")
    _evaluate_classification(fit_knn, embeddings, labels, data_name="train(KNN)")
    _evaluate_classification(fit_svm, embeddings, labels, data_name="train(SVM)")
    _evaluate_classification(fit_nn, embeddings, labels, data_name="train(NN)")

    # Eval on validation
    val_embeddings, val_labels = trainer.compute_all_embeddings(val_loader)
    val_embeddings = normalize(val_embeddings)
    logger.info("Evaluating classification on val data...")
    _evaluate_classification(fit_knn, val_embeddings, val_labels, data_name="val(KNN)")
    _evaluate_classification(fit_svm, val_embeddings, val_labels, data_name="val(SVM)")
    _evaluate_classification(fit_nn, val_embeddings, val_labels, data_name="val(NN)")

    if args.test:
        test_meta_df = pd.read_csv(TEST_META_PATH)
        test_dataset = LesionClassificationDataset(
            IMAGES_PATH, test_meta_df, transform=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=hparams.batch_size, num_workers=num_workers
        )
        embeddings, labels = trainer.compute_all_embeddings(test_loader)
        embeddings = normalize(embeddings)

        logger.info("Evaluating classification on test data...")
        _evaluate_classification(fit_knn, embeddings, labels, data_name="test(KNN)")
        _evaluate_classification(fit_svm, embeddings, labels, data_name="test(SVM)")
        _evaluate_classification(fit_nn, embeddings, labels, data_name="test(NN)")

    total_script_time = time.time() - script_start_time
    logger.info("Script done in %d seconds.", total_script_time)


def _evaluate_classification(
    model: ClassifierMixin,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    data_name: str = "train",
    minimal: bool = False,
) -> float:
    """Generate classification report

    Returns:
        AUC score
    """
    predictions = model.predict(embeddings)
    proba = model.predict_proba(embeddings)
    report = classification_report(labels, predictions)
    auc = roc_auc_score(labels, proba[:, 1])
    if not minimal:
        logger.info("%s data report:\n%s\nauc: %f\n", data_name, report, auc)
        RocCurveDisplay.from_predictions(labels, proba[:, 1])
        plt.savefig(PLOTS_PATH / f"{data_name}_roc")
    return auc


def _margin_weight(arr: np.ndarray, hparams: HyperParams):
    """Weight function which accepts neighbours that are within designated margin"""
    closest = arr.argmin(axis=1)
    weights = (arr <= hparams.margin).astype(int)
    weights[:, closest] = 1

    return weights


if __name__ == "__main__":
    main()
