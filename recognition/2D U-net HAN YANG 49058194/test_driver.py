"""
test_driver.py
--------------
Automated testing for the trained U-Net model using test data.

Input:
    - Trained model weights and test dataset.

Output:
    - Evaluation metrics such as Dice score displayed.

Usage:
    Run this script to evaluate the performance of the model on the test set.

Author: Han Yang
Date: 01/10/2024
"""
import os
import torch
from torch.utils.data import DataLoader
from modules import UNet  
from dataset import ProstateMRIDataset 
from train import train_model  
from predict import predict_and_evaluate 
from download import download_and_extract, load_and_process_nii_files  

# Configure hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_URL = "https://filesender.aarnet.edu.au/download.php?token=76f406fd-f55d-497a-a2ae-48767c8acea2&files_ids=23102543"

# Step 1: Download and prepare the data
root_dir = 'HipMRI_study_keras_slices_data'
processed_dir = os.path.join(root_dir, 'processed_nii_files') 

if not os.path.exists(root_dir):
    print("Downloading and preparing data...")
    download_and_extract(DATA_URL, root_dir)  

# Step 2: Process the NII file into NPY format
if not os.path.exists(processed_dir):
    print("Processing NIfTI files into npy format...")
    load_and_process_nii_files(root_dir, processed_dir, target_size=(128, 128))  

# Step 3: Train the model
print("Starting model training...")
train_model(processed_dir, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE) 

# Step 4: Prediction and Evaluation
print("Starting model prediction and evaluation...")
dice = predict_and_evaluate(processed_dir)  

# Step 5: Output Results
if dice >= 0.75:
    print(f"Model achieved the desired Dice score of 0.75 or above: {dice:.2f}")
else:
    print(f"Model did not achieve the desired Dice score: {dice:.2f}")
