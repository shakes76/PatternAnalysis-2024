"""
Make prediction and evaluate the model performance on test set.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ADNIDataset, ADNIDatasetTest
from utils import get_transform, set_seed
from modules import GFNet
from functools import partial
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/home/lcz/PatternAnalysis-2024/data/ADNI/AD_NC', type=str)
parser.add_argument('--show_progress', default="True", type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--device', default="cuda", type=str)
parser.add_argument('--test_seed', default=0, type=int)

args = parser.parse_args()

# fix training seed and define some global variables
set_seed(args.test_seed)
device = args.device if torch.cuda.is_available() else "cpu"
batch_size = args.batch_size
disable_tqdm = not (args.show_progress == "True")

script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, 'logs/GFNet')

if __name__ == "__main__":
    
    # load the dataset
    test_dataset = ADNIDatasetTest(root=args.data_path, transform=get_transform(train=False))

    # create model
    model = GFNet(img_size=210, in_chans=1, patch_size=14, embed_dim=384, depth=12, mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)


    model.load_state_dict(torch.load(os.path.join(log_dir, "best_gfnet.pt")))

    model.eval()
    preds_list = []
    true_list = []

    # test
    for data, labels in tqdm(test_dataset, disable=disable_tqdm):
        data, labels = data.to(device), labels.float().to(device)
        
        outputs = model(data)
        
        outputs = (torch.sigmoid(outputs) >= 0.5).float()
        preds = 1 if torch.mean(outputs.float()) >= 0.5 else 0 

        preds_list.append(preds)
        true_list.append(labels.item())
        
    preds_list = np.array(preds_list)
    true_list = np.array(true_list)

    # Calculate accuracy
    accuracy = accuracy_score(true_list, preds_list)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_list, preds_list)
    
    # Calculate precision
    precision = precision_score(true_list, preds_list)
    
    # Calculate recall
    recall = recall_score(true_list, preds_list)
    
    # Calculate F1 score
    f1 = f1_score(true_list, preds_list)

    # Print results
    print("Confusion Matrix:")
    print(f"TN\tFP\n{conf_matrix[0, 0]}\t{conf_matrix[0, 1]}")
    print(f"FN\tTP\n{conf_matrix[1, 0]}\t{conf_matrix[1, 1]}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")