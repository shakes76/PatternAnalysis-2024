"""
predict.py

Example usage of the trained vision transformer.

Author: Chiao-Yu Wang (Student No. 48007506)
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from modules import GFNet
from dataset import load_data
from constants import MODEL_SAVE_PATH

def predict(load_path, test_loader):
    # Load the trained model
    model = GFNet(num_classes=2)
    model.load_state_dict(torch.load(load_path))
    model.eval()

    y_true = []
    y_pred = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.float(), labels
            y_true.extend(labels.cpu().numpy())             # Collect true labels
            outputs = model(images)
            predicted = F.softmax(outputs, dim=1)           # Apply softmax to get probabilities
            y_pred.extend(predicted.argmax(dim=1).numpy())  # Collect predicted labels

    # Convert lists to numpy arrays for confusion matrix
    labels_true = np.array(y_true)
    labels_pred = np.array(y_pred)

    # Compute confusion matrix
    matrix = confusion_matrix(labels_true, labels_pred)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(x=j, y=i, s=matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actual Label', fontsize=18)
    plt.suptitle('Confusion Matrix', fontsize=18)
    plt.savefig('confusion_matrix.png')  # Save the confusion matrix as a PNG file
    plt.clf()


if __name__ == '__main__':
    # Load the dataset
    train_loader, val_loader, test_loader = load_data()

    # Call the predict function
    predict(MODEL_SAVE_PATH, test_loader)
