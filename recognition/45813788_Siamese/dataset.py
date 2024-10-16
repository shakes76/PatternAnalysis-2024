'''preprocess dataset'''
import pandas as pd
import torch
from torch.utils.data import DataLoader, dataloader
import os

current_dir = os.getcwd()
excel = os.path.join(current_dir,'recognition','45813788_Siamese','dataset', 'train-metadata.csv')

df = pd.read_csv(excel)

df= df.drop(columns=['Unnamed: 0'])
df= df.drop(columns=['patient_id'])

benign = df[df['target'] == 0].reset_index(drop=True)
malignant = df[df['target'] == 1].reset_index(drop=True)


print(malignant)
print(benign)




#print(malignant)
#print(benign)

# Check if the path exists