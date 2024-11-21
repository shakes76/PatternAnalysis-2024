"""Example usage of triamese network for skin lesion classification"""

import logging
import argparse

import pandas as pd
from sklearn import preprocessing, svm

from trainer import SiameseController, HyperParams
from dataset import load_single_image
import utils

logger = logging.getLogger(__name__)


def main():
    """Runs program"""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("-l", "--load-model")
    args = parser.parse_args()

    trainer = SiameseController(HyperParams(), "prediction_model")

    logger.info("Loading model...")
    if args.load_model is not None:
        trainer.load_model(args.load_model)
    else:
        trainer.load_model("best")

    logger.info("Computing image embeddings...")
    train_meta_df = pd.read_csv("data/train.csv")
    _, svm_train_dataloader = utils.get_classifier_loader(
        train_meta_df, batch_size=128, num_workers=2
    )
    embeddings, labels = trainer.compute_all_embeddings(svm_train_dataloader)
    embeddings = preprocessing.normalize(embeddings)

    logger.info("Training SVM")
    svc = svm.SVC(probability=True, random_state=42)
    svc = svc.fit(embeddings, labels)

    image = load_single_image(args.image_path)
    embedding = trainer.compute_embedding(image)
    embedding = preprocessing.normalize(embedding)

    prediction = svc.predict(embedding)[0]
    label_mapping = {0: "benign", 1: "malignant"}

    logger.info(
        "The lesion in the image is predicted to be %s", label_mapping[prediction]
    )


if __name__ == "__main__":
    main()
