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
device = torch.device("mps") #change to cuda for pc 
data_dir = '/Users/gghollyd/comp3710/report/AD_NC/'
model_path = '/Users/gghollyd/comp3710/report/module_weights.pth'

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

def compare_images(test_image, predicted_label, true_label, class_names):
    """
    Display the test image and its predicted label.
    """
    to_pil = ToPILImage()

    # Convert tensor to PIL image
    test_image_pil = to_pil(test_image.squeeze(0))  

    # Create figure to display the comparison
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Display the test image
    axs[0].imshow(test_image_pil, cmap='gray')
    axs[0].set_title(f'Test Image (True: {class_names[true_label]})')
    axs[0].axis('off')

    # Display the predicted image label
    axs[1].imshow(test_image_pil, cmap='gray')  # Display the same test image for visual comparison
    axs[1].set_title(f'Predicted Label: {class_names[predicted_label]}')
    axs[1].axis('off')

    # Show the comparison
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Set up model
    train_loader, val_loader, test_loader, class_names = dataloader(data_dir)
    model = VisionTransformer(
                                num_layers=8,
                                img_size=(240, 256),  
                                emb_size=768,         
                                patch_size=64,       
                                num_head=6,           
                                num_class=10        
                            ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    #train_model(model, train_loader, val_loader, optimizer, criterion, 2)
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set the model to evaluation mode
    # Run inference on the test set
    y_true, y_pred, test_images = run_inference(model, test_loader, device)

    for i in range(5):
        compare_images(test_images[i], y_pred[i], y_true[i], class_names)

    test_accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Calculate metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['NC', 'AD']))

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes=['NC', 'AD'])
