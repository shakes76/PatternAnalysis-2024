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
    cm = confusion_matrix(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    
    overall_accuracy = (y_true == y_pred).mean()
    
    return cm, precision, recall, f1, balanced_acc, auc_roc, overall_accuracy

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_prob):
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
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='coolwarm')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of embeddings')
    plt.savefig('tsne_plot.png')
    plt.close()

def main():
    # Hyperparameters
    batch_size = 32
    embedding_dim = 256
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
    cm, precision, recall, f1, balanced_acc, auc_roc, overall_accuracy = calculate_metrics(true_labels, predictions, probabilities)

    # Log results
    class_names = ['Benign', 'Malignant']
    logging.info(f"Overall Accuracy: {overall_accuracy:.4f}")
    logging.info(f"Balanced Accuracy: {balanced_acc:.4f}")
    logging.info(f"AUC-ROC: {auc_roc:.4f}")
    for i, class_name in enumerate(class_names):
        logging.info(f"{class_name} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-score: {f1[i]:.4f}")

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