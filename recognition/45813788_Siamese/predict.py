import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from dataset import ISICDataset
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report


def test(siamese, classifier, test_loader, images):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    siamese.to(device)
    siamese.eval()
    
    classifier.to(device)
    classifier.eval()

    all_preds = []
    all_labels = []
    all_probs = []


    with torch.no_grad():
        for images,labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            embeddings = siamese(images)
            output = classifier(embeddings).squeeze()

            preds = (output >= 0.5).float()
            probs = torch.sigmoid(output)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().detach().numpy())
            

    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute AUROC
    try:
        auroc = roc_auc_score(all_labels, all_probs)
        print(f"Test AUROC: {auroc:.4f}")
        print("Testing Classification Report:")
        print(classification_report(all_labels, all_preds, zero_division=0))
    except ValueError:
        auroc = float('nan')
        print("AUROC is undefined for the current test set.")


    try:
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'AUROC = {auroc:.4f}')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

    except ValueError:
        print("Could not plot ROC curve due to insufficient data.")

    accuracy = (all_preds == all_labels).mean()
    print(f"Overall Test Accuracy: {accuracy:.4f}")

    # Print confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
            
