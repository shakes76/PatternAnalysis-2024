import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from data import generate_dataloaders
from model import Siamese
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import yaml
import seaborn as sns

# Run prediction inference
def predict(model, dataloader_test, device):
    # Preparations, set model in evaluation mode
    model.eval()
    running_loss = 0
    record_label = []
    record_probpos = []
    record_predictions = []
    
    with torch.no_grad():
        for i, (anchor, label, _, _) in enumerate(dataloader_test):
            # Send to pytorch device
            anchor = anchor.to(device)
            out = model.classify(anchor)
            
            # Evaluate results
            probs = torch.softmax(out, dim=1)[:, 1] 
            _, preds = out.max(1)
            
            # Store evaluations
            record_predictions.extend(preds.cpu().numpy())
            record_probpos.extend(probs.cpu().numpy())
            record_label.extend(label.cpu().numpy())
    
    return np.array(record_predictions), np.array(record_probpos), np.array(record_label)

# Evaluate inference performance
def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    _, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    
    overall_accuracy = (y_true == y_pred).mean()
    
    return cm, recall, balanced_acc, auc_roc, overall_accuracy

# Display confusion matrix
def plot_confusion_matrix(cm, class_labels):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    plt.close()

# Display ROC curve
def plot_roc_curve(y_true, y_prob):

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)
    
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig("roc.png")
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
    model_path = data["ModelPath"]
    
    return batch_size, output_dir, model_path, train_ratio

def main():
    # Device configuration
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    
    print("PyTorch Version:", torch.__version__)
    print("PyTorch Device:", device_name)

    # Load params
    batch_size, output_dir, model_path, train_ratio = load_params()
    
    # Load data
    print("Loading data...")
    _, val_loader = generate_dataloaders(dir=output_dir, ratio=train_ratio, batch_size=batch_size)

    # Load model and best weights
    model = Siamese().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Prediction
    pred, probpos, labels = predict(model, val_loader, device)

    # Calculate metrics
    cm, recall, balanced_acc, auc_roc, overall_accuracy = calculate_metrics(labels, pred, probpos)

    # Display results
    class_names = ["Benign", "Malignant"]
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    # Display and show metrics
    plot_confusion_matrix(cm, class_names)
    plot_roc_curve(labels, probpos)

    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = recall[1]  # Same as recall for the positive class (Malignant)
    
    print(f"Specificity: {specificity:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")

if __name__ == "__main__":
    main()