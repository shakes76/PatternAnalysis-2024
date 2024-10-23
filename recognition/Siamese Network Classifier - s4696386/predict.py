"""
Provides an entry point to create, train, test, and evaluate the Neural Networks defined in modules.py

@author Kai Graz
"""
import matplotlib.pyplot as plt
import torch, os
from train import run_model, MALIGNANT, BENIGN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score

DEFAULT_BATCHSIZE = 16
DEFAULT_NUM_OF_EPOCHS = 32
DEFAULT_LEARNING_RATE = 0.0002
DEFAULT_THRESHOLD = 0.5
CLASS_NAMES = ["Benign", "Malignant"]
DEFAULT_SAVE_LOCATION = os.getcwd()

"""
AI and MT tools used to
 - Explain and provide examples for displaying confusion matrices
 - Suggest a cmap
"""


# Imitating train.test_classifier but more verbose
def evaluate_classifier(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_preds = []

    criterion = model.loss_criterion

    with torch.no_grad():
        # Feed data into model
        for (images, targets) in test_loader:
            images, targets = images.to(device), targets.to(device).float()
            outputs = model(images).squeeze()
            test_loss += criterion(outputs, targets).sum().item()

            # Store true and predicted values
            all_targets.extend(targets.cpu().numpy())
            pred = torch.where(outputs > DEFAULT_THRESHOLD, MALIGNANT, BENIGN)
            all_preds.extend(pred.cpu().numpy())
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)

    print(f"Classifier: Average loss: {test_loss:.4f}\nAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)")
    print(f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}\n")

    plot_confusion_matrix(all_preds, all_targets, CLASS_NAMES)

def plot_confusion_matrix(preds, targets, class_names=None):
    # Compute confusion matrix
    cm = confusion_matrix(targets, preds)
    
    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

def main():
    # Model passing (instead of saving and loading) allowed under https://edstem.org/au/courses/18266/discussion/2319727
    model, device, test_loader_classifier = run_model(DEFAULT_BATCHSIZE, DEFAULT_NUM_OF_EPOCHS, DEFAULT_LEARNING_RATE,
        save_model=True, test_verbose=True, train_verbose=True, save_location=DEFAULT_SAVE_LOCATION)

    evaluate_classifier(model, device, test_loader_classifier)

# Only run main when running file directly (not during imports)
if __name__ == "__main__":
    main()