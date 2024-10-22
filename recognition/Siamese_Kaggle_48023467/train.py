import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from data import generate_dataloaders
from model import Siamese
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import yaml

def run_epoch(model, dataloader_train, loss_triplet, loss_classify, optimizer, scaler, torch_device, torch_device_name):
    # Preparations, set model in training mode
    model.train()
    running_loss = 0
    record_label = []
    record_probpos = []
    record_predictions = []
    
    for i, (anchor, label, positive, negative) in enumerate(dataloader_train):
        # Send to pytorch device
        anchor, positive, negative, label = anchor.to(torch_device), positive.to(torch_device), negative.to(torch_device), label.to(torch_device)
        optimizer.zero_grad()

        # Main loop, obtain network outputs and evaluate loss
        with autocast():
            out_anchor, out_positive, out_negative = model(anchor, positive, negative)
            out_loss_triplet = loss_triplet(out_anchor, out_positive, out_negative)
            
            out_classifier = model.classify(out_anchor)
            out_loss_classifier = loss_classify(out_classifier, label)
            
            total_loss = out_loss_triplet + out_loss_classifier
        
        # Scale loss and apply learning
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update metric progress
        running_loss += total_loss.item()
        prob_pos = torch.softmax(out_classifier, dim=1)[:, 1]
        _, predictions = torch.max(out_classifier, 1)
        
        record_label.extend(label.cpu().numpy())
        record_probpos.extend(prob_pos.detach().cpu().numpy())
        record_predictions.extend(predictions.cpu().numpy())
        
        # Display running metric
        running_acc = accuracy_score(record_label, record_predictions)
        print(f"Loss: {(running_loss / (i + 1)):.4f}, Acc: {running_acc:.4f}")
            
    avg_loss = running_loss / len(dataloader_train)
    acc = accuracy_score(record_label, record_predictions)
    auc_roc = roc_auc_score(record_label, record_probpos)
    
    return avg_loss, acc, auc_roc
    
def run_validate(model, dataloader_val, loss_triplet, loss_classify, torch_device):
    # Preparations, set model in evaluation mode
    model.eval()
    running_loss = 0
    record_label = []
    record_probpos = []
    record_predictions = []
    
    print("Validating...")
    
    # No gradient calculations for evaluation
    with torch.no_grad():
        for i, (anchor, label, positive, negative) in enumerate(dataloader_val):
            # Send to pytorch device
            anchor, positive, negative, label = anchor.to(torch_device), positive.to(torch_device), negative.to(torch_device), label.to(torch_device)
            
            # Main loop, obtain network outputs and evaluate loss
            out_anchor, out_positive, out_negative = model(anchor, positive, negative)
            out_loss_triplet = loss_triplet(out_anchor, out_positive, out_negative)
            
            out_classifier = model.classify(out_anchor)
            out_loss_classifier = loss_classify(out_classifier, label)
            
            total_loss = out_loss_triplet + out_loss_classifier
            
            # Update metric progress
            running_loss += total_loss.item()
            prob_pos = torch.softmax(out_classifier, dim=1)[:, 1]
            _, predictions = torch.max(out_classifier, 1)
            
            record_label.extend(label.cpu().numpy())
            record_probpos.extend(prob_pos.detach().cpu().numpy())
            record_predictions.extend(predictions.cpu().numpy())
            
            # Display running metric
            running_acc = accuracy_score(record_label, record_predictions)
            print(f"Loss: {(running_loss / (i + 1)):.4f}, Acc: {running_acc:.4f}")
                
        avg_loss = running_loss / len(dataloader_val)
        acc = accuracy_score(record_label, record_predictions)
        auc_roc = roc_auc_score(record_label, record_probpos)
    
    return avg_loss, acc, auc_roc

def show_metrics(loss_train, loss_val, acc_train, acc_val, auc_train, auc_val):
    print("Displaying results:")
    
    fig, axs = plt.subplots(3, 2)
    
    # Training and validation loss
    axs[0, 0].plot(range(1, len(loss_train)+1), loss_train, label="Training Loss")
    axs[0, 1].plot(range(1, len(loss_val)+1), loss_val, label="Validation Loss")
    axs.flat[0].set(xlabel="Epoch", ylabel="Loss")
    axs.flat[1].set(xlabel="Epoch", ylabel="Loss")
    axs[0, 0].set_title("Training Loss")
    axs[0, 1].set_title("Validation Loss")

    # Training and validation classification accuracy
    axs[1, 0].plot(range(1, len(acc_train)+1), acc_train, label="Training Accuracy")
    axs[1, 1].plot(range(1, len(acc_val)+1), acc_val, label="Validation Accuracy")
    axs.flat[2].set(xlabel="Epoch", ylabel="Accuracy")
    axs.flat[3].set(xlabel="Epoch", ylabel="Accuracy")
    axs[1, 0].set_title("Training Accuracy")
    axs[1, 1].set_title("Validation Accuracy")

    # Training and validation AUC-ROC
    axs[2, 0].plot(range(1, len(auc_train)+1), auc_train, label="Training AUC-ROC")
    axs[2, 1].plot(range(1, len(auc_val)+1), auc_val, label="Validation AUC-ROC")
    axs.flat[4].set(xlabel="Epoch", ylabel="AUC-ROC")
    axs.flat[5].set(xlabel="Epoch", ylabel="AUC-ROC")
    axs[2, 0].set_title("Training AUC-ROC")
    axs[2, 1].set_title("Validation AUC-ROC")

    plt.tight_layout()
    plt.savefig("metrics.png")
    plt.close()
    
def load_params():
    with open("config.yaml", 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data_dir = data["DatasetImageDir"]
    label_csv = data["LabelCSV"]
    output_dir = data["OutputDir"]
    train_ratio = data["TrainTestRario"]
    oversample_ratio = data["OversampleRatio"]
    initial_run = data["FirstRun"]
    batch_size = data["BatchSize"]
    dropout_rate = data["ModelDropoutRate"]
    triplet_margin = data["TripletLossMargin"]
    lr = data["LearningRate"]
    n_epochs = data["EpochCount"]
    
    return data_dir, train_ratio, batch_size, triplet_margin, lr, n_epochs
    
def main():
    # Device configuration
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    
    print("PyTorch Version:", torch.__version__)
    print("PyTorch Device:", device_name)
    
    # Load params
    data_dir, train_ratio, batch_size, triplet_margin, lr, n_epochs = load_params()
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = generate_dataloaders(data_dir, ratio=train_ratio, batch_size=batch_size)
    
    # Load model and loss functions
    model = Siamese().to(device)
    triplet_loss = model.loss(margin=triplet_margin).to(device)
    classifier_loss = nn.CrossEntropyLoss().to(device)
    
    # Define other tools, filtering optimizer gradients for restraining parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    scaler = GradScaler()
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_aucs, val_aucs = [], []
    best_auc = 0
    
    # Run epochs
    for epoch in range(n_epochs):
        print(f"Running training epoch {epoch+1}")
        # Train
        train_loss, train_acc, train_auc = run_epoch(model, train_loader, triplet_loss, classifier_loss, optimizer, scaler, device, device_name)
        val_loss, val_acc, val_auc = run_validate(model, val_loader, triplet_loss, classifier_loss, device)
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC-ROC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC-ROC: {val_auc:.4f}")

        # Update LR
        scheduler.step(val_auc)

        # Update best model if higher accuracy reached
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best.pth')

    print("Training complete")
    
    # Display results
    show_metrics(train_losses, val_losses, train_accs, val_accs, train_aucs, val_aucs)
    return

if __name__ == "__main__":
    main()