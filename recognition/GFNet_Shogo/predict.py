"""
Showing example usage of your trained model. Print out any results and / or provide visualisations where applicable

Created by: Shogo Terashima
"""
import torch
from dataset import TestPreprocessing
from modules import GFNet
from dataset import TrainPreprocessing, TestPreprocessing
import os
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
learning_rate = 4.740668664000694e-05
weight_decay = 5.366743960698246e-05
drop_rate = 0.20676852620489247
drop_path_rate = 0.1987960305289435
batch_size = 128

#lr_4.740668664000694e-05_wd_5.366743960698246e-05_drop_0.20676852620489247_droppath_0.1987960305289435_bs_128/best_model.pt

# Load test data
test_dataset_path = "../dataset/AD_NC/test"
#test_dataset_path = "/home/groups/comp3710/ADNI/AD_NC/test"
test_data = TestPreprocessing(test_dataset_path, batch_size=batch_size)
test_loader = test_data.get_test_loader()


model = GFNet(
    img_size=224, 
    num_classes=1,
    blocks_per_stage=[3, 3, 27, 3], 
    stage_dims=[96, 192, 384, 768], 
    drop_rate=drop_rate,
    drop_path_rate=drop_path_rate,
    init_values=1e-6
)
model.to(device)
# Load model
checkpoint_path = f"./experiments/lr_{learning_rate}_wd_{weight_decay}_drop_{drop_rate}_droppath_{drop_path_rate}_bs_{batch_size}/best_model.pt"
csv_path = f"./experiments/lr_{learning_rate}_wd_{weight_decay}_drop_{drop_rate}_droppath_{drop_path_rate}_bs_{batch_size}//loss_log.csv"

model.load_state_dict(torch.load(checkpoint_path, weights_only = True))

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        # Forward pass
        outputs = model(inputs)
        preds = torch.sigmoid(outputs)
        
        # Convert predictions to binary values (0 or 1)
        predicted = (preds > 0.5).int()
        
        # Update the total number of samples
        total += labels.size(0)
        
        # Update the number of correct predictions
        correct += (predicted == labels.int()).sum().item()

# Calculate accuracy
accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")


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