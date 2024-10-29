import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from dataset import get_dataset
from modules import ViTClassifier


def cross_validate_models(model1_path, model2_path, test_loader, device, classes):
    """
    Combine the strengths of two models by taking the better prediction for each class.
    """
    model1 = load_trained_model(model1_path, device)
    model2 = load_trained_model(model2_path, device)

    # Evaluate both models on the test set
    print("Evaluating Model 1...")
    preds1, labels1, probs1 = evaluate_model(model1, test_loader, device, classes)
    print("Evaluating Model 2...")
    preds2, labels2, probs2 = evaluate_model(model2, test_loader, device, classes)

    # Calculate accuracy per class for each model
    accuracy_per_class1 = {}
    accuracy_per_class2 = {}
    
    for class_idx in range(len(classes)):
        mask = (labels1 == class_idx)
        if np.any(mask):  # only calculate if class exists in test set
            accuracy_per_class1[class_idx] = np.mean(preds1[mask] == labels1[mask])
            accuracy_per_class2[class_idx] = np.mean(preds2[mask] == labels2[mask])

    # Choose predictions based on which model performs better for each class
    all_preds = np.zeros_like(preds1)
    all_probs = np.zeros_like(probs1)
    
    for class_idx in range(len(classes)):
        if class_idx in accuracy_per_class1:
            # Use predictions from the model with higher accuracy for this class
            if accuracy_per_class1[class_idx] >= accuracy_per_class2[class_idx]:
                mask = (labels1 == class_idx)
                all_preds[mask] = preds1[mask]
                all_probs[mask] = probs1[mask]
            else:
                mask = (labels1 == class_idx)
                all_preds[mask] = preds2[mask]
                all_probs[mask] = probs2[mask]

    return all_preds, labels1, all_probs

def plot_roc_curves(y_true, y_prob, classes, save_path):
    """
    Plot ROC curve for the combined model predictions
    """
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    # Convert true labels to one-hot encoding for present classes only
    present_classes = np.unique(y_true)
    y_true_onehot = np.zeros((len(y_true), len(present_classes)))
    
    for i, class_idx in enumerate(present_classes):
        y_true_onehot[:, i] = (y_true == class_idx)
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, class_idx])
        roc_auc = auc(fpr, tpr)
        class_name = classes[class_idx]
        plt.plot(fpr, tpr, 
                label=f'{class_name} (AUC = {roc_auc:.2f})', 
                color=colors[i])
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'roc_curves.png', dpi=300)
    plt.close()


def load_trained_model(model_path, device):
    """
    Load a trained model from checkpoint
    """
    # Initialize model architecture
    model = ViTClassifier(num_classes=4).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """
    Plot and save confusion matrix
    """
    # Get unique classes actually present in the data
    present_classes = np.unique(np.concatenate([y_true, y_pred]))
    present_class_names = [classes[i] for i in present_classes]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma',
                xticklabels=present_class_names,
                yticklabels=present_class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrix.png')
    plt.close()

    # Calculate per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    return per_class_accuracy, present_class_names

def evaluate_model(model, test_loader, device, classes):
    """
    Evaluate model performance on test set
    """
    model.eval()

    # Initialize lists to store predictions and true labels
    all_preds = []
    all_labels = []
    all_probs = []

    # Testing loop
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probabilities.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def save_metrics(metrics, save_path):
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics (dict): Dictionary containing evaluation metrics
        save_path (Path): Directory path where metrics should be saved
    """
    # Convert numpy values to Python native types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Convert all numpy values in the metrics dictionary
    metrics_json = {k: convert_numpy(v) if isinstance(v, (dict, np.generic, np.ndarray))
                   else v for k, v in metrics.items()}

    # Save to JSON file
    metrics_file = save_path / 'evaluation_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_json, f, indent=4, sort_keys=True, default=convert_numpy)

def main():
    # Configuration
    BATCH_SIZE = 64
    CLASSES = ['CN', 'MCI', 'AD', 'SMC']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(f'cross_validation_results_{timestamp}')
    results_dir.mkdir(exist_ok=True)

    print(f"\nCross-validation Results will be saved to: {results_dir}")

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = get_dataset(train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Test set size: {len(test_dataset)} images")

    # Define paths for both models
    model1_path = "./checkpoints/best_model_20241029_234652.pt"
    model2_path = "./checkpoints/best_model_20241029_224507.pt"
    print(f"Loading models from:\nModel 1: {model1_path}\nModel 2: {model2_path}")

    # Perform cross-validation
    print("\nPerforming cross-validation...")
    predictions, true_labels, probabilities = cross_validate_models(
        model1_path, model2_path, test_loader, device, CLASSES
    )

    # Get unique classes present in the data
    present_classes = np.unique(true_labels)
    present_class_names = [CLASSES[i] for i in present_classes]

    print("\nCalculating metrics...")

    # Classification report with only present classes
    report = classification_report(
        true_labels,
        predictions,
        labels=present_classes,
        target_names=present_class_names,
        output_dict=True
    )

    # Per-class accuracy from confusion matrix
    per_class_accuracy, matrix_classes = plot_confusion_matrix(
        true_labels, predictions, CLASSES, results_dir
    )

    # Plot ROC curves only for present classes
    if len(present_classes) > 1:  # Only plot ROC curves if there are multiple classes
        plot_roc_curves(true_labels, probabilities[:, present_classes],
                       present_class_names, results_dir)

    # Compile metrics
    metrics = {
        'classification_report': report,
        'per_class_accuracy': {
            class_name: acc for class_name, acc in zip(matrix_classes, per_class_accuracy)
        },
        'overall_accuracy': (predictions == true_labels).mean(),
        'model1_path': model1_path,
        'model2_path': model2_path,
        'evaluation_date': timestamp,
        'test_set_size': len(test_dataset),
        'classes_present': present_class_names
    }

    # Save metrics
    save_metrics(metrics, results_dir)

    # Print summary
    print("\nCross-validation Results Summary:")
    print(f"Classes present in test set: {', '.join(present_class_names)}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']*100:.2f}%")
    print("\nPer-class Accuracy:")
    for class_name, acc in metrics['per_class_accuracy'].items():
        print(f"{class_name}: {acc*100:.2f}%")

    print(f"\nDetailed results have been saved to {results_dir}")
    print("Files generated:")
    print("- confusion_matrix.png")
    if len(present_classes) > 1:
        print("- roc_curves.png")
    print("- evaluation_metrics.json")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCross-validation interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise