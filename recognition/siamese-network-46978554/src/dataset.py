"""Data loader for loading and preprocessing data"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("../data")


def create_train_test_split(data_dir: Path, seed: int = 42, train_split: float = 0.8):
    import os
    import shutil

    rng = np.random.default_rng(seed)

    # Create train/test split in subdirectory so original data isn't affected
    out_dir = data_dir / "train-test-split"

    os.makedirs(out_dir / "train", exist_ok=True)
    os.makedirs(out_dir / "test", exist_ok=True)

    metadata = pd.read_csv(data_dir / "train-metadata.csv")

    # Shuffle data and create train/test split
    idxs = np.arange(len(metadata))
    rng.shuffle(idxs)
    cutoff = int(len(metadata) * train_split)
    train_idxs = idxs[:cutoff]
    test_idxs = idxs[cutoff:]

    def create_split(idxs: np.ndarray, split_name: str):
        md_split = metadata.iloc[idxs]

        md_split_benign = md_split[md_split["target"] == 0]
        md_split_malign = md_split[md_split["target"] == 1]

        md = pd.concat(
            [
                _create_positive_pairs(md_split_benign, seed),
                _create_positive_pairs(md_split_malign, seed),
                _create_negative_pairs(md_split_malign, md_split_benign, seed),
            ]
        )

        md = md.sample(frac=1, random_state=seed).reset_index(drop=True)
        md.to_csv(out_dir / f"{split_name}-metadata.csv", index=False)

        for isic_id in md_split["isic_id"]:
            src = data_dir / f"train-image/image/{isic_id}.jpg"
            dest = out_dir / f"{split_name}/{isic_id}.jpg"
            shutil.move(src, dest)

    create_split(train_idxs, "train")
    create_split(test_idxs, "test")


def _create_positive_pairs(md: pd.DataFrame, seed: int = 42):
    rng = np.random.default_rng(seed)

    idxs = np.arange(len(md))
    rng.shuffle(idxs)
    cutoff = len(idxs) // 2

    pairs = dict()
    pairs["isic_id1"] = md["isic_id"].iloc[:cutoff].reset_index(drop=True)
    pairs["isic_id2"] = md["isic_id"].iloc[cutoff:].reset_index(drop=True)
    pairs["target"] = [0] * (len(idxs) // 2)  # positive pairs ==> label is 0
    pairs = pd.DataFrame.from_dict(pairs)

    return pairs


def _create_negative_pairs(md1: pd.DataFrame, md2: pd.DataFrame, seed: int = 42):
    rng = np.random.default_rng(seed)

    # Match half of data in the larger class to smaller class
    # md1 assumed to be smaller class
    md2_idxs = np.arange(len(md2) // 2)
    md1_idxs = rng.choice(np.arange(len(md1)), len(md2_idxs))

    pairs = dict()
    pairs["isic_id1"] = md1["isic_id"].iloc[md1_idxs].reset_index(drop=True)
    pairs["isic_id2"] = md2["isic_id"].iloc[md2_idxs].reset_index(drop=True)
    pairs["target"] = [1] * (len(md2) // 2)  # negative pairs ==> label is 1
    pairs = pd.DataFrame.from_dict(pairs)

    return pairs
