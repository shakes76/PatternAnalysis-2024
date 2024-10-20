mport torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from modules import *
import matplotlib.pyplot as plt
from dataset import get_data_loaders
from train import validate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GFNet(
            img_size=512, 
            patch_size=16, embed_dim=512, depth=19, mlp_ratio=4, drop_rate=0.1
        )
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
model.to(device)


zip_path = "ADNI_AD_NC_2D.zip"
extract_to = "data"
train_loader, val_loader, test_loader = get_data_loaders(zip_path, extract_to, batch_size=32, train_split = 0.85)


# Load the best model for testing
model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_accuracy = validate(model, test_loader, criterion, device, is_draw=True)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")