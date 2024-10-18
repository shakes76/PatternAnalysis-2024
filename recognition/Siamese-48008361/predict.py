"""
predict.py

This module contains functions for predicting and evaluating the performance of a
trained Siamese Network model on the ISIC 2020 skin lesion dataset. It includes
functions for making predictions, calculating various metrics, and visualizing results.

Author: Zain Al-Saffi
Date: 18th October 2024
"""

import os
import torch
import torch.nn as nn
from dataset import get_data_loaders
from modules import get_model
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict(model, test_loader, device):
    """
    Make predictions using the trained model on the test dataset.

    Args:
        model (nn.Module): The trained Siamese Network model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        tuple: Tuple containing arrays of predictions, probabilities, true labels, and embeddings.
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    all_embeddings = []
    
    with torch.no_grad():
        for batch_idx, (anchor, _, _, labels) in enumerate(tqdm(test_loader, desc="Predicting")):
            anchor = anchor.to(device)
            embeddings = model.get_embedding(anchor)
            outputs = model.classify(anchor)
            # Probability of positive class (to be used for ROC curve)
            probs = torch.softmax(outputs, dim=1)[:, 1] 
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels), np.concatenate(all_embeddings)

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate various performance metrics for the model predictions.

    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.
        y_prob (np.array): Predicted probabilities for the positive class.

    Returns:
        tuple: Tuple containing confusion matrix, recall,
               balanced accuracy, AUC-ROC, and overall accuracy.
    """
    cm = confusion_matrix(y_true, y_pred)
    _, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    
    overall_accuracy = (y_true == y_pred).mean()
    
    return cm, recall, balanced_acc, auc_roc, overall_accuracy

def plot_confusion_matrix(cm, class_names):
    """
    Plot and save the confusion matrix.

    Args:
        cm (np.array): Confusion matrix.
        class_names (list): List of class names.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_prob):
    """
    Plot and save the ROC curve.

    Args:
        y_true (np.array): True labels.
        y_prob (np.array): Predicted probabilities for the positive class.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC-ROC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

def visualize_tsne(embeddings, labels):
    """
    Visualize the embeddings using t-SNE and save the plot.

    Args:
        embeddings (np.array): Embeddings from the Siamese Network.
        labels (np.array): True labels for the embeddings.
    """
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='coolwarm')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of embeddings')
    plt.savefig('tsne_plot.png')
    plt.close()

def main():
    """
    Main function to run the prediction and evaluation pipeline.
    """
    # Hyperparameters
    batch_size = 32
    embedding_dim = 320
    data_dir = 'preprocessed_data/'
    model_path = 'best_model.pth'

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
    predictions, probabilities, true_labels, embeddings = predict(model, test_loader, device)

    # Calculate metrics
    cm, recall, balanced_acc, auc_roc, overall_accuracy = calculate_metrics(true_labels, predictions, probabilities)

    # Log results
    class_names = ['Benign', 'Malignant']
    logging.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logging.info(f"Balanced Accuracy: {balanced_acc:.4f}")
    logging.info(f"AUC-ROC: {auc_roc:.4f}")


    # Plot and save confusion matrix
    plot_confusion_matrix(cm, class_names)
    logging.info("Confusion matrix saved as 'confusion_matrix.png'")

    # Plot and save ROC curve
    plot_roc_curve(true_labels, probabilities)
    logging.info("ROC curve saved as 'roc_curve.png'")

    # Visualize embeddings using t-SNE
    visualize_tsne(embeddings, true_labels)
    logging.info("t-SNE plot saved as 'tsne_plot.png'")

    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = recall[1]  # Same as recall for the positive class (Malignant)
    
    logging.info(f"Specificity: {specificity:.4f}")
    logging.info(f"Sensitivity: {sensitivity:.4f}")

if __name__ == '__main__':
    main()