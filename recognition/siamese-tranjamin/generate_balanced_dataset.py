import pandas as pd
import shutil
import os

try:
    shutil.rmtree("datasets/balanced")
except:
    pass
os.mkdir("datasets/balanced")
os.mkdir("datasets/balanced/positive")
os.mkdir("datasets/balanced/negative")

labels = pd.read_csv("datasets/train_labels.csv")
dups = list(pd.read_csv("datasets/ISIC_2020_Training_Duplicates.csv")["image_name_2"])
positive_labels = labels.loc[labels["target"] == 1]
negative_labels = labels.loc[labels["target"] == 0]

count = 0
for filename in positive_labels["image_name"]:
    if filename in dups:
        continue

    count += 1
    shutil.copy(f"datasets/images/{filename}.jpg", "datasets/balanced/positive")

ncount = 0
for filename in negative_labels["image_name"]:
    if ncount == count:
        break
    if filename in dups:
        continue

    ncount += 1
    shutil.copy(f"datasets/images/{filename}.jpg", "datasets/balanced/negative")
