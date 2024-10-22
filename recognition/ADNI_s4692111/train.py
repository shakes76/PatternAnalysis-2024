import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules import ADNICNN
from dataset import get_data_loaders
import os

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the test and train dataset
    train_loader, test_loader = get_data_loaders("recognition/ADNI_s4692111/AD_NC", batch_size=16)

    model = ADNICNN().to(device)
    
    # check is there have saved model
    model_path = 'adni_cnn_model.pth'
    if os.path.exists(model_path):
        print(f"Load saved model：{model_path}")
        model.load_state_dict(torch.load(model_path))
    else:
        print("Did not find saved model, start a new training.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # start train
    model.train()
    num_epochs = 1
    bar = tqdm(range(num_epochs * len(train_loader)), desc="train")

    # loop train
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
            
            bar.update()

        scheduler.step()

        # print the lost of epoch.
        tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
        torch.save(model.state_dict(), 'adni_cnn_model.pth')
    return model,test_loader
