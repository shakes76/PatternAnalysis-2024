import torch
import torch.nn as nn
from modules import VisionTransformer
from dataset import dataloader
from train import train_model
import torch.optim as optim
import numpy
import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torchvision.transforms import ToPILImage
from torchvision import datasets, transforms
from PIL import Image

device = torch.device("cuda") #change to cuda for pc 
data_dir = '/Users/georg/OneDrive/Documents/comp3710/report/AD_NC/'
model_path = '/Users/georg/OneDrive/Documents/comp3710/module_weights.pth'


def run_inference(model, test_loader, device):
    """
    Run inference on the test set.
    """
    model.to(device)
    all_preds = []
    all_labels = []
    all_test_images = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) 
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_test_images.extend(inputs.cpu()) 

    return all_labels, all_preds, all_test_images

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def check_prediction(image_path, model, true_label, class_names, device):
    # Load and preprocess the image
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((240, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # Assuming normalization used in training
    ])
    image = transform(image).unsqueeze(0).to(device) # Add batch dimension and move to device

    # Set the model to evaluation mode and make prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted_label = torch.max(output, 1)
        predicted_label = predicted_label.item()

    # Check if the prediction matches the true label
    is_correct = predicted_label == true_label
    predicted_label_name = class_names[predicted_label]

    return is_correct, predicted_label_name


if __name__ == '__main__':
    # Set up model
    train_loader, val_loader, test_loader, class_names = dataloader(data_dir)
    model = VisionTransformer(
                                num_layers=8,
                                img_size=(240, 256),  
                                emb_size=768,         
                                patch_size=16,       
                                num_head=6,           
                                num_class=10        
                            ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, val_loader, optimizer, criterion, 15)
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  
    # Run inference on the test set
    y_true, y_pred, test_images = run_inference(model, test_loader, device)

    image_path = '/Users/georg/OneDrive/Documents/comp3710/report/AD_NC/test/AD/388206_85.jpeg'
    true_label = 1 # Assume this is'AD'

    # Assuming model is already loaded and on the correct device
    is_correct, predicted_label_name = check_prediction(image_path, model, true_label, class_names, device)

    if is_correct:
        print(f"The model correctly predicted {predicted_label_name}.")
    else:
        print(f"The model incorrectly predicted {predicted_label_name}.")

    test_accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Calculate metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['NC', 'AD']))

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes=['NC', 'AD'])
