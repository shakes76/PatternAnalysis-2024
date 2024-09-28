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

    metadata = pd.read_csv(args.input)

    benign = metadata[metadata[TARGET] == 0]
    malignant = metadata[metadata[TARGET] == 1]

    benign_train, benign_test = _split_sub_population(benign, IMAGE_NAME)
    benign_train, benign_val = _split_sub_population(benign_train, IMAGE_NAME)
    malignant_train, malignant_test = _split_sub_population(malignant, IMAGE_NAME)
    malignant_train, malignant_val = _split_sub_population(malignant_train, IMAGE_NAME)
    _summarise_num_pairs(benign_train, malignant_train)
    
    # Shuffle dfs
    benign_train = benign_train.sample(frac=1, random_state=42)
    malignant_train = malignant_train.sample(frac=1, random_state=42)

    malignant_pairs = list(itertools.combinations(malignant_train[IMAGE_NAME], 2))

    # Collect benign pairs
    benign_pairs = []
    for pair in itertools.combinations(benign_train, 2):
        benign_pairs.append(pair)

        if len(benign_pairs) == len(malignant_pairs):
            break
    
    # Collect negative pairs
    negative_pairs = []
    for pair in itertools.product(benign_train, malignant_train):
        negative_pairs.append(pair)

        if len(negative_pairs) == len(malignant_pairs):
            break

    all_pairs = malignant_pairs + benign_pairs + negative_pairs

    image_ones = [image_one for image_one, _ in all_pairs]
    image_twos = [image_two for _, image_two in all_pairs]

    pairs_df = pd.DataFrame({"image_one": image_ones, "image_two": image_twos})
    pairs_df.to_csv(pathlib.Path("data/pairs.csv"), index=False)



def _split_sub_population(df: pd.DataFrame, id_name: str) -> tuple:
    df_train = df.sample(random_state=42, frac=0.8)
    df_test = df[~df[id_name].isin(df_train[id_name])]

    return df_train, df_test

def _summarise_num_pairs(benign_df: pd.DataFrame, malignant_df: pd.DataFrame) -> None:
    pos_benign = math.comb(len(benign_df), 2)
    pos_malig = math.comb(len(malignant_df), 2)
    neg = len(benign_df) * len(malignant_df)
    logger.info(f"Num positive benign pairs {pos_benign}\n" 
          f"Num positive malignant pairs {pos_malig}\n" 
          f"Num negative pairs {neg}") 

if __name__ == "__main__":
    main()