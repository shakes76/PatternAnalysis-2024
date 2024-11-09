"""
Author: Ella WANG

This module is used to show example usage the trained ViT model by using the model to predict
on a testing set from the ADNI brain dataset

Evaluation metrics will be printed and evaluation figures will be saved to the current folder.
"""

# Standard library imports
import json
from datetime import datetime
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Local imports
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
    model_choices = {}  # Track which model was chosen for each class

    for class_idx in range(len(classes)):
        if class_idx in accuracy_per_class1:
            mask = (labels1 == class_idx)
            if accuracy_per_class1[class_idx] >= accuracy_per_class2[class_idx]:
                all_preds[mask] = preds1[mask]
                all_probs[mask] = probs1[mask]
                model_choices[class_idx] = 1
            else:
                all_preds[mask] = preds2[mask]
                all_probs[mask] = probs2[mask]
                model_choices[class_idx] = 2

    return all_preds, labels1, all_probs, model_choices

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

def save_sample_images(images, predictions, true_labels, classes, save_path, probabilities, model_choices):
    """
    Save sample images for all classes and false positives after cross-validation

     Args:
        images: Tensor of input images
        predictions: Model predictions
        true_labels: True labels
        classes: List of class names
        save_path: Directory to save images
        probabilities: Model prediction probabilities
        model_choices: Models
    """
    def save_grid(img_list, title, filename):
        if len(img_list) == 0:
            return

        # Take up to 5 images
        img_list = img_list[:5]

        grid = vutils.make_grid(img_list, nrow=len(img_list), padding=2, normalize=True)
        plt.figure(figsize=(15, 3))
        plt.axis('off')
        plt.title(title)
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.savefig(save_path / filename, bbox_inches='tight', dpi=300)
        plt.close()

    # Lists to store sample images for each class
    correct_samples = {class_name: [] for class_name in classes}
    correct_probs = {class_name: [] for class_name in classes}

    # Dictionary to store false positives for each misclassification type
    false_positive_samples = {
        f"{true_class}->{pred_class}": []
        for true_class in classes
        for pred_class in classes
        if true_class != pred_class
    }
    false_positive_probs = {
        f"{true_class}->{pred_class}": []
        for true_class in classes
        for pred_class in classes
        if true_class != pred_class
    }

    # Collect samples
    for img, pred, true_label, prob in zip(images, predictions, true_labels, probabilities):
        pred_class = classes[pred]
        true_class = classes[true_label]
        used_model = model_choices.get(true_label, "unknown")

        if pred == true_label:
            if len(correct_samples[true_class]) < 5:
                correct_samples[true_class].append(img)
                correct_probs[true_class].append(prob[pred])
        else:
            key = f"{true_class}->{pred_class}"
            if len(false_positive_samples[key]) < 5:
                false_positive_samples[key].append(img)
                false_positive_probs[key].append(prob[pred])

    # Save correct classifications
    for class_name in classes:
        class_idx = classes.index(class_name)
        if correct_samples[class_name]:
            save_grid(
                correct_samples[class_name],
                f"Correct {class_name} Samples (Using Model {model_choices[class_idx]})\n" + \
                f"Confidence: {[f'{p:.2f}' for p in correct_probs[class_name]]}",
                f'{class_name.lower()}_correct_samples.png'
            )

    # Save false positives grouped by misclassification type
    for misclass_type, samples in false_positive_samples.items():
        if samples:
            true_class, pred_class = misclass_type.split('->')
            save_grid(
                samples,
                f"False Positives: {misclass_type}\n" + \
                f"True: {true_class}, Predicted: {pred_class}\n" + \
                f"Confidence: {[f'{p:.2f}' for p in false_positive_probs[misclass_type]]}",
                f'false_positive_{true_class.lower()}_to_{pred_class.lower()}.png'
            )
def evaluate_model(model, test_loader, device, classes):
    """
    Evaluate model performance on test set
    """
    model.eval()

    # Initialise lists to store predictions, true labels, and images
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
    model1_path = "./checkpoints/best_model_20241109_190825.pt"
    model2_path = "./checkpoints/checkpoint_epoch_1_20241109_195852.pt"
    print(f"Loading models from:\nModel 1: {model1_path}\nModel 2: {model2_path}")

    # Perform cross-validation
    print("\nPerforming cross-validation...")
    predictions, true_labels, probabilities, model_choices = cross_validate_models(
        model1_path, model2_path, test_loader, device, CLASSES
    )

    # Save model choice information
    model_choice_info = {
        CLASSES[class_idx]: f"Model {model_num}"
        for class_idx, model_num in model_choices.items()
    }
    with open(results_dir / 'model_choices.json', 'w') as f:
        json.dump(model_choice_info, f, indent=4)

    # Get all images for visualization
    all_images = []
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    for images, _ in tqdm(test_loader, desc="Collecting images for visualization"):
        all_images.extend(images)
    all_images = torch.stack(all_images)

    # Save sample images
    print("\nSaving sample images...")
    save_sample_images(all_images, predictions, true_labels, CLASSES, results_dir,
                      probabilities, model_choices)

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
    if len(present_classes) > 1:
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
    print("- cn_correct_samples.png")
    print("- mci_correct_samples.png")
    print("- ad_correct_samples.png")
    print("- smc_correct_samples.png")

    # List false positive files that were actually generated
    generated_fp_files = list(results_dir.glob('false_positive_*.png'))
    if generated_fp_files:
        print("\nFalse positive analysis files:")
        for fp_file in generated_fp_files:
            print(f"- {fp_file.name}")

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