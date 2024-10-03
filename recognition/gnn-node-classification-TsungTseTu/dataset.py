import numpy as np

def load_facebook_data():
    dataset_path = "./data/facebook.npz"
    data = np.load(dataset_path)
    print(data.files)
    return data


# Run the function to inspect the dataset
if __name__ == '__main__':
    load_facebook_data()
