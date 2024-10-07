import pandas as pd
import shutil
import os

shutil.rmtree("datasets/balanced")
os.mkdir("datasets/balanced")
os.mkdir("datasets/balanced/positive")
os.mkdir("datasets/balanced/negative")

labels = pd.read_csv("datasets/train_labels.csv")
positive_labels = labels.loc[labels["target"] == 1]
negative_labels = labels.loc[labels["target"] == 0]

for filename in positive_labels["image_name"]:
    shutil.copy(f"datasets/images/{filename}.jpg", "datasets/balanced/positive")

for ind, filename in enumerate(negative_labels["image_name"]):
    if ind == len(positive_labels):
        break
    shutil.copy(f"datasets/images/{filename}.jpg", "datasets/balanced/negative")
