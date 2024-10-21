"""
Contains the data loader for loading and preprocessing the ISIC 2020 Data
"""

import os
import numpy as np
import pandas as pd


def get_isic2020_data(metadata_path, image_dir, data_subset):
    """
    Returns: images, labels
    """
    metadata = pd.read_csv(metadata_path)
    
    # Add the file extension to isic_id to match image filenames
    metadata['image_file'] = metadata['isic_id'] + '.jpg'
    
    # Map image filename to target class and get a list of the file paths
    image_to_label = dict(zip(metadata['image_file'], metadata['target']))
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img in image_to_label]

    # If we are using subset of the data, ensure that we try our best to have an equal number of each class
    pos_paths = [img for img in image_paths if image_to_label[os.path.basename(img)] == 1][:data_subset // 2]
    neg_paths = [img for img in image_paths if image_to_label[os.path.basename(img)] == 0][:data_subset // 2]
    image_paths = pos_paths + neg_paths

    # Get a list of the labels for the images we are using
    labels = [image_to_label[os.path.basename(path)] for path in image_paths]
    return np.array(image_paths), np.array(labels)