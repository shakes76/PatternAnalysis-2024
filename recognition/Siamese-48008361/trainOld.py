import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from dataset import get_data_loaders
from modules import get_model, get_loss
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_epoch(model, train_loader, triplet_loss, classifier_loss, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (anchor, positive, negative, labels) in enumerate(pbar):
        anchor, positive, negative, labels = anchor.to(device), positive.to(device), negative.to(device), labels.to(device)

        with autocast():
            anchor_out, positive_out, negative_out = model(anchor, positive, negative)
            triplet_loss_val = triplet_loss(anchor_out, positive_out, negative_out)
            classifier_out = model.classify(anchor)
            classifier_loss_val = classifier_loss(classifier_out, labels)
            loss = 0.7 * triplet_loss_val + 0.3 *classifier_loss_val

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_loss += loss.item()
        probs = torch.softmax(classifier_out, dim=1)[:, 1]  # Probability of positive class
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

        pbar.set_postfix({'Loss': running_loss / (batch_idx + 1)})

    avg_loss = running_loss / len(train_loader)
    auc_roc = roc_auc_score(all_labels, all_probs)

    return avg_loss, auc_roc

def validate(model, val_loader, triplet_loss, classifier_loss, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []

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
            probs = torch.softmax(classifier_out, dim=1)[:, 1]
            _, preds = classifier_out.max(1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            pbar.set_postfix({'Loss': running_loss / (batch_idx + 1)})

    avg_loss = running_loss / len(val_loader)
    auc_roc = roc_auc_score(all_labels, all_probs)

    # Generate classification report
    # report = classification_report(all_labels, all_preds, target_names=['benign', 'malignant'], digits=4)
    # logging.info(f"\nValidation Classification Report:\n{report}")

    return avg_loss, auc_roc

def plot_loss_auc(train_losses, val_losses, train_aucs, val_aucs):
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot AUC-ROC
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_aucs)+1), train_aucs, label='Train AUC-ROC')
    plt.plot(range(1, len(val_aucs)+1), val_aucs, label='Validation AUC-ROC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC-ROC')
    plt.legend()
    plt.title('Training and Validation AUC-ROC')

    plt.tight_layout()
    plt.savefig('training_plots.png')
    plt.close()
    logging.info("Training plots saved as 'training_plots.png'")

def main():
    # Hyperparameters
    batch_size = 32
    embedding_dim = 128
    learning_rate = 3e-4
    weight_decay = 5e-4  # Weight decay parameter
    num_epochs = 20
    data_dir = 'preprocessed_data/'

    # Early stopping threshold for validation AUC-ROC
    early_stopping_threshold = 0.85
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_loader, val_loader = get_data_loaders(data_dir=data_dir, batch_size=batch_size, split_ratio=0.8, num_workers=6)
    logging.info("Data loaded successfully")

    model = get_model(embedding_dim=embedding_dim).to(device)
    triplet_loss = get_loss(margin=1.0).to(device)
    classifier_loss = nn.CrossEntropyLoss().to(device)
    
    #Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=learning_rate, 
                            weight_decay=weight_decay)

    # # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=2, verbose=True)

    

    # Reinitialize scheduler
    #lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    scaler = GradScaler()

    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []

    best_val_auc = 0
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")

        # if epoch == 5:
        #     for param in model.feature_extractor.parameters():
        #         param.requires_grad = True
        #     # Reinitialize optimizer with all parameters
        #     optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #     # Reinitialize scheduler
        #     lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

        train_loss, train_auc = train_epoch(model, train_loader, triplet_loss, classifier_loss, optimizer, device, scaler)
        val_loss, val_auc = validate(model, val_loader, triplet_loss, classifier_loss, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        logging.info(f"Train Loss: {train_loss:.4f}, Train AUC-ROC: {train_auc:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}, Val AUC-ROC: {val_auc:.4f}")

        # Step the learning rate scheduler
        lr_scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f"New best model saved with validation AUC-ROC: {best_val_auc:.4f}")

        if val_auc >= early_stopping_threshold and epoch > 10:
            logging.info(f"Early stopping triggered. Validation AUC-ROC {val_auc:.4f} exceeds threshold of {early_stopping_threshold}")
            break

    logging.info("Training completed")

    plot_loss_auc(train_losses, val_losses, train_aucs, val_aucs)

    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_auc = validate(model, val_loader, triplet_loss, classifier_loss, device)
    logging.info(f"Test Loss: {test_loss:.4f}, Test AUC-ROC: {test_auc:.4f}")

    # Generate confusion matrix (you may want to adjust the threshold for binary classification)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for anchor, _, _, labels in val_loader:
            anchor = anchor.to(device)
            outputs = model.classify(anchor)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            # Threshold for binary classification
            preds = (probs > 0.5).float() 
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['benign', 'malignant'], yticklabels=['benign', 'malignant'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    logging.info("Confusion matrix saved as 'confusion_matrix.png'")

if __name__ == '__main__':
    main()