"""
Load test set, the best model, and CSV file that recorded train and validation loss across each epoch obtained by training.
Predict the test set using the stored model and measure and report the accuracy. 
It also plots the trend of train loss and validation loss at each epoch from the csv file saved by train.py.

Created by:     Shogo Terashima
ID:             S47779628
Last update:    24/10/2024
"""
import torch
from dataset import TestPreprocessing
from modules import GFNet
from dataset import TrainPreprocessing, TestPreprocessing, CombinedPreprocessing
import os
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Set the hyperparameter (for identifying the folder that the model is saved by train.py)
learning_rate = 0.005876215431947011
weight_decay = 0.0007870757149220836
drop_rate = 0.10125914358064875
drop_path_rate = 1.0
batch_size = 64
warmup_epochs = 5
t_max = 30

# Load test data
train_dataset_path = "../dataset/AD_NC/train"
test_dataset_path = "../dataset/AD_NC/test"
#test_dataset_path = "/home/groups/comp3710/ADNI/AD_NC/test"
#train_dataset_path = "/home/groups/comp3710/ADNI/AD_NC/train"

seed = 20 #! make sure use the same seed as the train.py
data_preprocessor = CombinedPreprocessing(
    train_path=train_dataset_path,
    test_path=test_dataset_path,
    batch_size=batch_size,
    num_workers=1,
    val_split=0.2,
    seed=seed
)
train_loader, val_loader, test_loader = data_preprocessor.get_data_loaders()

# Define the model (GFNet-H-B)
model = GFNet(
    image_size=224, 
    num_classes=1,
    blocks_per_stage=[3, 3, 27, 3], 
    stage_dims=[96, 192, 384, 768], 
    drop_rate=drop_rate,
    drop_path_rate=drop_path_rate,
    init_values=1e-6
)
model.to(device)

# model path and csv path
checkpoint_path = f"./experiments4/lr_{learning_rate}_wd_{weight_decay}_drop_{drop_rate}_droppath_{drop_path_rate}_bs_{batch_size}/best_model.pt"
csv_path = f"./experiments4/lr_{learning_rate}_wd_{weight_decay}_drop_{drop_rate}_droppath_{drop_path_rate}_bs_{batch_size}//loss_log.csv"

# load the model
model.load_state_dict(torch.load(checkpoint_path, weights_only = True))

model.eval()
all_corrects = []
all_predictions = []

with torch.no_grad():
    for inputs, correct in test_loader: # using test loader
        inputs = inputs.to(device)
        correct =  correct.to(device).float()
        output = model(inputs)
        predicted = torch.round(torch.sigmoid(output)).squeeze(1)
        all_predictions.extend(predicted.cpu().numpy())
        all_corrects.extend(correct.cpu().numpy())

# Calculate accuracy using scikit-learn
acc = accuracy_score(all_corrects, all_predictions) * 100
print(f'Accuracy on test set: {acc:.2f}%')


# Read the data from the CSV file
df = pd.read_csv(csv_path)

# Plotting the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss', marker='o')
plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss', marker='s')

# Adding labels, title, and legend
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid()

# Show the plot
plt.show()