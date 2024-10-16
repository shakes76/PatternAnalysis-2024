"""
Showing example usage of your trained model. Print out any results and / or provide visualisations where applicable

Created by: Shogo Terashima
"""
import torch
from dataset import TestPreprocessing
from modules import GFNet
from sklearn.metrics import accuracy_score
from dataset import TrainPreprocessing, TestPreprocessing
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001
weight_decay = 0.05
drop_rate = 0.05
drop_path_rate = 0.05
batch_size = 128


# Load test data
#test_dataset_path = "../dataset/AD_NC/test"
test_dataset_path = "/home/groups/comp3710/ADNI/AD_NC/test"
test_data = TestPreprocessing(test_dataset_path, batch_size=batch_size)
test_loader = test_data.get_test_loader()


model = GFNet(
    img_size=224, 
    num_classes=1,
    initial_embed_dim=32, 
    blocks_per_stage=[1, 1, 1, 1], 
    stage_dims=[32, 64, 128, 256], 
    drop_rate=drop_rate,
    drop_path_rate=drop_path_rate,
    init_values=1e-5
)
model.to(device)

# Load model
checkpoint_path = f"experiments/lr_{learning_rate}_wd_{weight_decay}_drop_{drop_rate}_droppath_{drop_path_rate}_bs_{batch_size}/best_model.pt"
model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        outputs = model(inputs)
        preds = torch.sigmoid(outputs)
        preds = preds > 0.5
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}")