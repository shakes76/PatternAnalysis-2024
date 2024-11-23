import torch
from modules import GFNet  
from dataset import test_loader  
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model = GFNet(num_classes=2).to(device)
model.load_state_dict(torch.load('./result/best_model.pth'))  
model.eval()  

# Define the function for prediction and evaluation
def predict_and_evaluate(model, test_loader):
    correct = 0
    total = 0
    all_preds = []  
    all_labels = []  

    with torch.no_grad():  
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Get predicted labels
            _, predicted = torch.max(outputs, 1)  
            
            # Append predictions to list
            all_preds.extend(predicted.cpu().numpy())  
            # Append true labels to list
            all_labels.extend(labels.cpu().numpy())  

            # Count correct predictions
            correct += (predicted == labels).sum().item()  
            # Count total samples
            total += labels.size(0)  

    # Calculate accuracy
    accuracy = correct / total
    print(f'Accuracy on test set: {accuracy * 100:.2f}%')

    # Generate and visualize the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)  # Create confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NC', 'AD'], yticklabels=['NC', 'AD'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig("./result/confusion_matrix.png")  
    plt.show()

# Run prediction and evaluation
predict_and_evaluate(model, test_loader)
