import pathlib
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = pathlib.Path("data")
ALL_META_DIR = DATA_DIR / "all.csv"

TARGET = "target"
IMAGE_COL = "image_name"

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    metadata = pd.read_csv(ALL_META_DIR)

    train, test = train_test_split(
        metadata, test_size=0.2, random_state=42, stratify=metadata[TARGET]
    )
    train, val = train_test_split(
        train, test_size=0.2, random_state=42, stratify=train[TARGET]
    )

    logger.info(
        "Size of sets\nall %d\ntrain %d\nval %d\ntest %d",
        len(metadata),
        len(train),
        len(val),
        len(test),
    )

    train.to_csv(pathlib.Path("data/train.csv"), index=False)
    val.to_csv(pathlib.Path("data/val.csv"), index=False)
    test.to_csv(pathlib.Path("data/test.csv"), index=False)


if __name__ == "__main__":
    main()
