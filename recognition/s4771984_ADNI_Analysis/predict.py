import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from dataset import get_data_loaders
from modules import GFNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Evaluation of the model
def evaluate_model(model, test_loader, criterion):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0

    with torch.no_grad():  
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    return all_preds, all_labels, avg_loss

# Plotting the confusion matrix
def plot_confusion_matrix(labels, preds):
    conf_matrix = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['NC', 'AD'], yticklabels=['NC', 'AD'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# For the classification report and saving the model
def main():
    # Load test dataset
    _, _, test_loader = get_data_loaders('/content/drive/MyDrive/ADNI/AD_NC/train', '/content/drive/MyDrive/ADNI/AD_NC/test')

    # Initialization of the model and load the trained model weights
    model = GFNet(num_classes=2).to(device)
    model.load_state_dict(torch.load('/content/drive/MyDrive/ADNI/saved_model.pth'))
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model on test dataset
    all_preds, all_labels, test_loss = evaluate_model(model, test_loader, criterion)

    print(f"Test Loss: {test_loss:.4f}")

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=['NC', 'AD'])
    print("Classification Report:\n", report)

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds)

if __name__ == '__main__':
    main()
