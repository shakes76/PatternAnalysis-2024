import torch
import torch.optim as optim
import torch.nn as nn
from modules import create_model
from dataset import get_dataloaders

def train_model(data_dir='/home/groups/comp3710/ADNI/AD_NC', batch_size=128, learning_rate=1e-4, num_classes=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders, dataset_sizes = get_dataloaders(data_dir, batch_size)
    model = create_model(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    best_acc = 0.0