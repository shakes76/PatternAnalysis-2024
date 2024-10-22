# dataset.py
# Currently it loads the smaller 2020 Kaggle dataset and visualises the distribution/ samples of classes
# Author: Harrison Martin

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_metadata(csv_path):
    """
    Load the metadata CSV file into a pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    return df


def visualise_class_distribution(df):
    """
    Visualise the distribution of classes in the dataset.
    """
    class_counts = df['target'].value_counts()
    class_counts.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=0)
    plt.show()


def show_sample_images(df, image_folder, num_samples=5):
    """
    Display a few sample images from the dataset.
    """
    sample_df = df.sample(n=num_samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for idx, (i, row) in enumerate(sample_df.iterrows()):
        img_path = os.path.join(image_folder, f"{row['isic_id']}.jpg")
        image = Image.open(img_path)
        axes[idx].imshow(image)
        axes[idx].axis('off')
        axes[idx].set_title(f"Label: {row['target']}")
    plt.show()

metadata_df = load_metadata('recognition/SiameseClassifier_46972691/test_dataset_2020_Kaggle/train-metadata.csv')

visualise_class_distribution(metadata_df)

show_sample_images(metadata_df, 'recognition/SiameseClassifier_46972691/test_dataset_2020_Kaggle/train-image/image', num_samples=5)