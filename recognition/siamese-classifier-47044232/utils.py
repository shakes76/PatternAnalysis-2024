"""
The helper methods used throughout.

Made by Joshua Deadman
"""

import math
import pandas as pd

from config import TESTING, VALIDATION

def split_data(label_path) -> tuple[list, list, list]:
    """ Splits the labels into a training, testing and validation set.
        The data is split with a 1:1 amount of benign and malignant data, omitting excess benign images.

    Arguments:
        label_path (str): The absolute path to .csv file holding labels.
    Returns:
        A tuple of lists holding dicts with the isic_id and target value.
        The order of lists is (Training, Testing, Validaiton).
    """
    df = pd.read_csv(label_path)
    malignant_count = sum(df["target"] == 1)
    malignant = df[df["target"] == 1]
    malignant = malignant.drop("patient_id", axis="columns")
    # Reduce benign samples to the same size as malignant
    benign = df.drop(labels=malignant.index.to_list()).sample(n=malignant_count)
    benign = benign.drop("patient_id", axis="columns")

    # Exclusive sets of malignant images
    test_m = malignant.sample(n=math.floor(malignant_count * (TESTING+VALIDATION)))
    val_m = test_m.sample(n=math.floor(malignant_count * VALIDATION))
    train_m = malignant.drop(labels=test_m.index.to_list())
    test_m = test_m.drop(labels=val_m.index.to_list())
    
    # Exclusive sets of benign data
    test_b = benign.sample(n=math.floor(malignant_count * (TESTING+VALIDATION)))
    val_b = test_b.sample(n=math.floor(malignant_count * VALIDATION))
    train_b = benign.drop(labels=test_b.index.to_list())
    test_b = test_b.drop(labels=val_b.index.to_list())

    # Sample used to randomise the order
    train = pd.concat([train_m, train_b]).sample(frac=1)
    test = pd.concat([test_m, test_b]).sample(frac=1)
    val = pd.concat([val_m, val_b]).sample(frac=1)

    return train.to_dict(orient="records"), test.to_dict(orient="records"), val.to_dict(orient="records")
