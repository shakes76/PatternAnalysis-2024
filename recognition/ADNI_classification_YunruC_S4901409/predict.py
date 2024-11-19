import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from modules import *
import matplotlib.pyplot as plt
from dataset import get_data_loaders
from train import validate

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The best hyperparameters found out of 10 trials
model = GFNet(
                img_size=512, 
                patch_size=16, embed_dim=512, depth=19, mlp_ratio=4, drop_path_rate=0.08789879211387502,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

#print(model)
criterion = nn.CrossEntropyLoss()
model.to(device)


zip_path = "ADNI_AD_NC_2D.zip"
extract_to = "data"
train_loader, val_loader, test_loader = get_data_loaders(zip_path, extract_to, batch_size=32, train_split = 0.85)


# Load the best model for testing
model.load_state_dict(torch.load('model.pth'))
test_loss, test_accuracy, cm = validate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

#Confusion matrix
print(f"Confusion Matrix:\n{cm}")


'''
# Hyperparameters with High Test Accuracy
model = GFNet(
                img_size=512, 
                patch_size=16, embed_dim=512, depth=19, mlp_ratio=4, drop_path_rate=0.17728764992362356,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
#print(model)
criterion = nn.CrossEntropyLoss()
model.to(device)


zip_path = "ADNI_AD_NC_2D.zip"
extract_to = "data"
train_loader, val_loader, test_loader = get_data_loaders(zip_path, extract_to, batch_size=32, train_split = 0.85)


# Load the best model for testing
model.load_state_dict(torch.load('model_2.pth'))
test_loss, test_accuracy, cm = validate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

#Confusion matrix
print(f"Confusion Matrix:\n{cm}")
'''
