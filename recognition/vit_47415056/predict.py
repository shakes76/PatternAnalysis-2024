import os
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from modules import initialize_model
from dataset import get_test_loader
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the folder structure if it does not exist
output_dir = "vit_47415056/graphs"
os.makedirs(output_dir, exist_ok=True)

def predict_and_visualize(model_path, test_data_dir):
    """
    Load a model, make predictions on test data, and visualize the results.

    Parameters:
    - model_path (str): Path to the saved model weights.
    - test_data_dir (str): Directory path for test data.
    """
    
    model = initialize_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = get_test_loader(test_data_dir)
    all_preds, all_labels = [], []

    print("\nTesting the model again on the test dataset...\n")
    for images, labels in test_loader:
        images = images.to(DEVICE)
        with torch.no_grad():
            predictions = model(images).argmax(dim=1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Confusion Matrix:\n{conf_matrix}\nTest Accuracy: {accuracy:.2%}")

    # Plotting the confusion matrix as a covariance matrix (heatmap) with numbers
    plt.figure()
    plt.matshow(conf_matrix, cmap='viridis')
    plt.title("Confusion Matrix (Covariance Matrix)")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Add numbers to each cell
    for (i, j), value in np.ndenumerate(conf_matrix):
        plt.text(j, i, f"{value}", ha="center", va="center", color="white")

    plt.savefig(os.path.join(output_dir, "covariance_matrix.png"))

if __name__ == "__main__":
    model_path = "model_weights.pth"
    test_data_dir = "/home/groups/comp3710/ADNI/AD_NC/test"
    predict_and_visualize(model_path, test_data_dir)