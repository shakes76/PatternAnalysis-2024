import numpy as np

def load_facebook_data():
    dataset_path = "./data/facebook.npz"
    data = np.load(dataset_path)
    return data


