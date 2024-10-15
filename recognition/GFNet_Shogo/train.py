import torch
from dataset import TrainPreprocessing, TestPreprocessing

# path to datasets
train_dataset_path = "../dataset/AD_NC/train"
test_dataset_path = "../dataset/AD_NC/test"

# Load train and validation data
train_data = TrainPreprocessing(train_dataset_path, batch_size=128)
train_loader, val_loader = train_data.get_train_val_loaders(val_split=0.2)

# Load test data
test_data = TestPreprocessing(test_dataset_path, batch_size=128)
test_loader = test_data.get_test_loader()

