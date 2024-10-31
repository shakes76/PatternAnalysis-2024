import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from modules import initialize_model
from dataset import get_test_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_and_visualize(model_path, test_data_dir):
    # Load the model
    model = initialize_model()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load test data
    test_loader = get_test_loader(test_data_dir)
    
    all_preds, all_labels = [], []
    print("\nTesting the model again on the test dataset...\n")
    for images, labels in test_loader:
        images = images.to(DEVICE)
        with torch.no_grad():
            predictions = model(images).argmax(dim=1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

    # Calculate metrics
    conf_matrix = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Confusion Matrix:\n{conf_matrix}\nTest Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    model_path = "model_weights.pth"
    test_data_dir = "/home/groups/comp3710/ADNI/AD_NC/test"
    predict_and_visualize(model_path, test_data_dir)