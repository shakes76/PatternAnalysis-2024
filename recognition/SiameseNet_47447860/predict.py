import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from siamese import SiameseNetwork
from libs.dataset import Dataset

if __name__ == "__main__":
    # The path to the model's checkpoint - where weights are saved ---------------------------
    checkpoint_path = ""

    # If we decide to save the data to the FileSystem rather than calling the data processing every time ---------------------
    data_path = ""

    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get the data here (as a tensor dataloader) ------------------------------------
    test_data = None

    criterion = torch.nn.BCELoss()

    checkpoint = torch.load(checkpoint_path)
    model = SiameseNetwork(backbone=checkpoint['backbone'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    losses = []
    correct = 0
    total = 0

    # Will have to change the format of the data to fit this for-loop structure
    for i, ((img1, img2), target, (class1, class2)) in enumerate(test_data):
        print("[{} / {}]".format(i, len(test_data)))

        img1, img2, target = map(lambda x: x.to(device), [img1, img2, target])
        class1 = class1[0]
        class2 = class2[0]

        similarity = model(img1, img2)
        loss = criterion(similarity, target)

        losses.append(loss.item())
        correct += torch.count_nonzero(target == (similarity > 0.5)).item()
        total += len(target)

        fig = plt.figure("class1={}\tclass2={}".format(class1, class2), figsize=(4, 2))
        plt.suptitle("cls1={}  conf={:.2f}  cls2={}".format(class1, similarity[0][0].item(), class2))

        # save the plot

    print("Validation: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses) / len(losses), correct / total))