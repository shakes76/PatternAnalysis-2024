import torch
import torch.nn as nn
import numpy as np
import os.path as osP
from torch.utils.data import DataLoader
from dataset import ADNI, ADNITest
from modules import GFNet
from functools import partial
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BATCH_SIZE = 32
MODEL_PATH = '/Users/rorymacleod/Desktop/Uni/sem 2 24/COMP3710/Report/gfnet.pt'


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    test_data = ADNITest()
    model = GFNet(in_channels=1, embed_dim=384)

    model.load_state_dict(torch.load(MODEL_PATH)).to(device)
    model.eval()
    
    preds = []
    true = []

    for data, labels in tqdm(test_data, disable=False):
        data, labels = data.to(device), labels.float().to(device)

        outputs = model(data)
        outputs = torch.sigmoid(outputs).mean().item()

        pred = 1 if outputs > 0.5 else 0

        preds.append(pred)
        true.append(labels.item())
    
    preds = np.array(preds)
    true = np.array(true)

    acc = accuracy_score(true, preds)
    conf_matrix = confusion_matrix(true, preds)
    precision = precision_score(true, preds)
    recall = recall_score(true, preds)
    f1 = f1_score(true, preds)

    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix")
    print(f"True Negative: {conf_matrix[0, 0]}")
    print(f"True Positive: {conf_matrix[1, 1]}")
    print(f"False Negative: {conf_matrix[1, 0]}")
    print(f"False Positive: {conf_matrix[0, 1]}")