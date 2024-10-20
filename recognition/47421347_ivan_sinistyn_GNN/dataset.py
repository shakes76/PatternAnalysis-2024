""" The functions in this file are loading and preprocessing
    the Facebook Page-Page Large Network dataset
"""

import numpy as np
import pandas as pd
import time
FILE_PATH = "./facebook.npz"


def load_data(filepath: str):

    data = np.load(filepath)

    target = data["target"]
    edges = data["edges"]
    features = data["features"]

    print()
    return data

if __name__ == "__main__":
    start_time = time.time()
    data = load_data(FILE_PATH)

    end_time = time.time()

    print(f"Time taken to load the data: {(end_time-start_time)} seconds")

    print(len(data["edges"]))
   