import torch
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from modules import initialize_model
from dataset import get_test_loader

# Set the device to GPU if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_and_visualize(model_path, test_data_dir):
    """
    Load a model, make predictions on test data, and visualize the results.

    Parameters:
    - model_path (str): Path to the saved model weights.
    - test_data_dir (str): Directory path for test data.
    """
    
    # Load the pre-trained model and set to evaluation mode
    model = initialize_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load test data using the custom DataLoader
    test_loader = get_test_loader(test_data_dir)
    
    all_preds, all_labels = [], []
    print("\nTesting the model again on the test dataset...\n")
    
    # Iterate over test data and make predictions
    for images, labels in test_loader:
        images = images.to(DEVICE)
        with torch.no_grad():
            predictions = model(images).argmax(dim=1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

    # Calculate and print confusion matrix and accuracy score
    conf_matrix = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Confusion Matrix:\n{conf_matrix}\nTest Accuracy: {accuracy:.2%}")

    # Plotting the confusion matrix for visual analysis
    plt.figure()
    plt.matshow(conf_matrix, cmap='viridis')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    # Define the model path and test data directory
    model_path = "model_weights.pth"
    test_data_dir = "/home/groups/comp3710/ADNI/AD_NC/test"
    predict_and_visualize(model_path, test_data_dir)