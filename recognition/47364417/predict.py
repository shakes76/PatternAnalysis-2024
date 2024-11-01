import os
import torch
from modules import create_model
from dataset import get_dataloaders
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def predict():
    """
    Loads a saved model (or trains a new one if none exists) and tests it on the testing data.
    """
    # Set device to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Define directories and model path.
    data_dir = '/home/groups/comp3710/ADNI/AD_NC'
    checkpoints_dir = 'checkpoints'
    final_model_path = os.path.join(checkpoints_dir, 'final_model.pth')

    # Get dataloaders and class names.
    dataloaders, class_names = get_dataloaders(data_dir)
    num_classes = len(class_names)
    print(f'Classes: {class_names}')

    # Initialize the model.
    model = create_model(num_classes)
    model = model.to(device)

    # Load the saved model if possible.
    if os.path.exists(final_model_path):
        print(f'Loading saved model from {final_model_path}')
        model.load_state_dict(torch.load(final_model_path, map_location=device))
    else:
        print('No saved model found. Training a new model...')
        from train import train_model
        train_model()
        model.load_state_dict(torch.load(final_model_path, map_location=device))

    # Test the model.
    model.eval()
    test_running_corrects = 0
    test_total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            test_running_corrects += torch.sum(preds == labels.data)
            test_total_samples += inputs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate testing accuracy.
    test_acc = test_running_corrects.double() / test_total_samples * 100
    print(f'Test Accuracy: {test_acc:.2f}%')

    # Generate confusion matrix.
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
