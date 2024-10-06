import os
import torch
import torch.nn as nn
from dataset import get_data_loaders
from modules import get_model
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (anchor, _, _, labels) in enumerate(tqdm(test_loader, desc="Predicting")):
            anchor = anchor.to(device)
            outputs = model.classify(anchor)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    overall_accuracy = (y_true == y_pred).mean()
    
    return cm, precision, recall, f1, balanced_acc, overall_accuracy

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()

def extract_data_from_log(log_file):
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Extract initial training accuracy for epoch 1
    initial_acc_match = re.search(r"Training: 100%.*?Acc=([\d.]+)", log_content)
    if initial_acc_match:
        train_accs.append(float(initial_acc_match.group(1)))
    
    for line in log_content.split('\n'):
        if "Train Loss:" in line:
            match = re.search(r"Train Loss: ([\d.]+), Train Acc: ([\d.]+)%", line)
            if match:
                train_losses.append(float(match.group(1)))
                train_accs.append(float(match.group(2)))
        elif "Val Loss:" in line:
            match = re.search(r"Val Loss: ([\d.]+), Val Acc: ([\d.]+)%", line)
            if match:
                val_losses.append(float(match.group(1)))
                val_accs.append(float(match.group(2)))
    
    return train_losses, val_losses, train_accs, val_accs

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close()

def plot_accuracies(train_accs, val_accs, test_acc):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Accuracy', marker='o')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Validation Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png')
    plt.close()

def main():
    # Hyperparameters
    batch_size = 32
    embedding_dim = 128
    data_dir = 'preprocessed_data/'
    model_path = 'best_model.pth'
    log_file = 'training.log'  # Update this to your log file name

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data loading
    _, test_loader = get_data_loaders(data_dir=data_dir, batch_size=batch_size)
    logging.info("Test data loaded successfully")

    # Model initialization and loading weights
    model = get_model(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    logging.info(f"Model loaded from {model_path}")

    # Predict
    predictions, true_labels = predict(model, test_loader, device)

    # Calculate metrics
    cm, precision, recall, f1, balanced_acc, overall_accuracy = calculate_metrics(true_labels, predictions)

    # Log results
    class_names = ['Benign', 'Malignant']
    logging.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logging.info(f"Balanced Accuracy: {balanced_acc:.4f}")
    for i, class_name in enumerate(class_names):
        logging.info(f"{class_name} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-score: {f1[i]:.4f}")

    # Plot and save confusion matrix
    plot_confusion_matrix(cm, class_names)
    logging.info("Confusion matrix saved as 'confusion_matrix.png'")

    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = recall[1]  # Same as recall for the positive class (Malignant)
    
    logging.info(f"Specificity: {specificity:.4f}")
    logging.info(f"Sensitivity: {sensitivity:.4f}")

    # Extract data from log file and create graphs
    train_losses, val_losses, train_accs, val_accs = extract_data_from_log(log_file)
    
    plot_losses(train_losses, val_losses)
    logging.info("Loss plot saved as 'loss_plot.png'")
    
    plot_accuracies(train_accs, val_accs, overall_accuracy)
    logging.info("Accuracy plot saved as 'accuracy_plot.png'")

if __name__ == '__main__':
    main()