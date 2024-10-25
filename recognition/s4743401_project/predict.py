import torch
import torch.nn as nn
from modules import VisionTransformer
from dataset import dataloader
from train import train_model
import torch.optim as optim
import numpy
import matplotlib as plt
from sklearn.metrics import confusion_matrix, classification_report
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

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get the predicted class
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

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

    train_model(model, train_loader, val_loader, optimizer, criterion, 2)
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set the model to evaluation mode
    # Run inference on the test set
    y_true, y_pred = run_inference(model, test_loader, device)

    # Calculate metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['NC', 'AD']))

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes=['NC', 'AD'])
