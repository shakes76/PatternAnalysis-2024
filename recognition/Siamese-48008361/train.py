import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import get_data_loaders
from modules import get_model, get_loss
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import datetime

# Setting up logging to see the loading bottlenecks
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_epoch(model, train_loader, triplet_loss, classifier_loss, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (anchor, positive, negative, labels) in enumerate(pbar):
        anchor, positive, negative, labels = anchor.to(device), positive.to(device), negative.to(device), labels.to(device)

        with autocast():
            anchor_out, positive_out, negative_out = model(anchor, positive, negative)
            triplet_loss_val = triplet_loss(anchor_out, positive_out, negative_out)
            classifier_out = model.classify(anchor)
            classifier_loss_val = classifier_loss(classifier_out, labels)
            loss = triplet_loss_val + classifier_loss_val

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_loss += loss.item()
        _, predicted = classifier_out.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'Loss': running_loss / (batch_idx + 1), 'Acc': 100. * correct / total})

    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, triplet_loss, classifier_loss, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for batch_idx, (anchor, positive, negative, labels) in enumerate(pbar):
            anchor, positive, negative, labels = anchor.to(device), positive.to(device), negative.to(device), labels.to(device)

            anchor_out, positive_out, negative_out = model(anchor, positive, negative)
            triplet_loss_val = triplet_loss(anchor_out, positive_out, negative_out)
            classifier_out = model.classify(anchor)
            classifier_loss_val = classifier_loss(classifier_out, labels)
            loss = triplet_loss_val + classifier_loss_val

            running_loss += loss.item()
            _, predicted = classifier_out.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'Loss': running_loss / (batch_idx + 1), 'Acc': 100. * correct / total})

    return running_loss / len(val_loader), 100. * correct / total

def main():
    # Hyperparameters
    batch_size = 32
    embedding_dim = 128
    learning_rate = 1e-3
    num_epochs = 10
    data_dir = 'preprocessed_data/'

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data loading
    train_loader, val_loader = get_data_loaders(data_dir=data_dir, batch_size=batch_size)
    logging.info("Data loaded successfully")

    # Model initialization
    model = get_model(embedding_dim=embedding_dim).to(device)
    triplet_loss = get_loss(margin=1.0).to(device)
    classifier_loss = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    scaler = GradScaler()

    # Training loop
    best_val_acc = 0
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, triplet_loss, classifier_loss, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, triplet_loss, classifier_loss, device)
        
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logging.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Learning rate scheduler step
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")

    logging.info("Training completed")

    

if __name__ == '__main__':
    main()