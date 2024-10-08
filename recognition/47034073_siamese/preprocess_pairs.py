"""Pre-process training metadata into pairs for training. Handle split for validation and test."""

import pathlib
import itertools
import math
import logging
import argparse

import pandas as pd

TARGET = "target"
IMAGE_NAME = "image_name"

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    args = parser.parse_args()

    if args.input is None:
        args.input = "data/train.csv"

    metadata = pd.read_csv(args.input)

    benign = metadata[metadata[TARGET] == 0]
    malignant = metadata[metadata[TARGET] == 1]

    benign_train, benign_test = _split_sub_population(benign, IMAGE_NAME)
    benign_train, benign_val = _split_sub_population(benign_train, IMAGE_NAME)
    malignant_train, malignant_test = _split_sub_population(malignant, IMAGE_NAME)
    malignant_train, malignant_val = _split_sub_population(malignant_train, IMAGE_NAME)
    _summarise_num_pairs(benign_train, malignant_train)

    train_df = pd.concat([benign_train, malignant_train])
    val_df = pd.concat([benign_val, malignant_val])
    test_df = pd.concat([benign_test, malignant_test])

    train_df.to_csv(pathlib.Path("data/train.csv"), index=False)
    val_df.to_csv(pathlib.Path("data/val.csv"), index=False)
    test_df.to_csv(pathlib.Path("data/test.csv"), index=False)


def _split_sub_population(df: pd.DataFrame, id_name: str) -> tuple:
    df_train = df.sample(random_state=42, frac=0.8)
    df_test = df[~df[id_name].isin(df_train[id_name])]

    return df_train, df_test


def _summarise_num_pairs(benign_df: pd.DataFrame, malignant_df: pd.DataFrame) -> None:
    pos_benign = math.comb(len(benign_df), 2)
    pos_malig = math.comb(len(malignant_df), 2)
    neg = len(benign_df) * len(malignant_df)
    logger.info(
        "Num positive benign pairs %d\n"
        "Num positive malignant pairs %d\n"
        "Num negative pairs %d",
        pos_benign,
        pos_malig,
        neg,
    )


if __name__ == "__main__":
    main()
