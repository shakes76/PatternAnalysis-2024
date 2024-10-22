import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path
import argparse
import torch.nn.functional as F  # Import this for applying softmax
from torch.nn import CrossEntropyLoss

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import pandas as pd
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

## Custom py files
from modules import ViT
from train import evaluate
from dataset import get_dataloaders 


# Function to display images and predictions
def imshow(img, label=None, prediction=None, font_size=8):
    """
    Unnormalizes the image and displays it.
    Args:
        img (torch.Tensor): The image tensor to display.
        label (str, optional): True label for the image.
        prediction (tuple, optional): A tuple with predicted label and its probability, e.g., (1, 0.75).
        font_size (int, optional): Font size for the text. Default is 8.
    """
    img = img.permute(1, 2, 0).numpy()  # Convert from [C, H, W] to [H, W, C]
    img = img * 0.2229 + 0.1156  # Undo normalization (using mean=0.1156, std=0.2229)
    img = img.clip(0, 1)  # Clip to valid range [0, 1]
    
    plt.imshow(img)
    
    # Construct the title with proper formatting
    title = f"True: {label}" if label is not None else ""
    if prediction is not None:
        pred_label, prob = prediction
        title += f", Pred: {pred_label} (Prob: {prob:.2f})"
        
    plt.title(title, fontsize=font_size)
    plt.axis('off')


def predict_batch(model, dataloader, device, num_images=32, font_size=8, idx_to_class={0: "AD", 1: "Normal"}):
    """
    Load a batch of images, make predictions, and display the images with predictions.
    
    Args:
        model (torch.nn.Module): The trained model for making predictions.
        dataloader (torch.utils.data.DataLoader): Dataloader with the images.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        num_images (int, optional): Number of images to display. Default is 32.
        font_size (int, optional): Font size for the text. Default is 8.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        images, labels = next(iter(dataloader))  # Get a batch of images and labels
        images, labels = images[:num_images].to(device), labels[:num_images].to(device)  # Limit to num_images
        
        outputs = model(images)  # Get model predictions (logits)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)
        
        # Get the predicted class index (and its corresponding probability)
        _, preds = torch.max(probabilities, 1)
        
        # Convert probabilities to CPU for display
        probabilities = probabilities.cpu().numpy()

        # Create a grid of images with predictions and their probabilities
        plt.figure(figsize=(16, 8))  # Adjust the figure size to be smaller (width x height)
        for i in range(num_images):
            plt.subplot(4, 8, i + 1)  # Adjust grid layout (4 rows, 8 columns)
            prob = probabilities[i][preds[i].item()]  # Get probability of the predicted class
            label_name = idx_to_class[labels[i].item()]  # Get the true label as a class name
            pred_name = idx_to_class[preds[i].item()]  # Get the predicted label as a class name
            imshow(images[i].cpu(), label=label_name, prediction=(pred_name, prob), font_size=font_size)
        plt.tight_layout()
        plt.savefig(f"plots/batch_predictions.png")  # Save the batch images with predictions
        print("Saved prediction images!!")



def evaluate_model(model, dataloader, device, idx_to_class={0: "AD", 1: "Normal"}):
    """
    Evaluates the model on the test set and generates a classification report.
    
    Args:
        model (torch.nn.Module): The trained model for evaluation.
        dataloader (torch.utils.data.DataLoader): Dataloader with the test data.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
    
    Returns:
        str: Classification report as a string.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # Get the predicted class index
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Generate classification report using the class names
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    class_report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print("Classification Report:\n", class_report)

    # Save classification report
    with open("plots/classification_report.txt", "w") as f:
        f.write(class_report)

    # Generate confusion matrix using the class names
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", conf_matrix)

    # Save confusion matrix as an image
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f"plots/confusion_matrix.png")
    print("Saved confusion matrix as png in /plots!")
    plt.close()

    # Calculate precision, recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    
    return class_report



def main():
    parser = argparse.ArgumentParser()

    # Add argument for model path
    parser.add_argument('--model_path', type=str, default="models/best_model.pth", help="Path to the trained model")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for prediction")
    parser.add_argument('--image_size', type=int, default=224, help="Image size for the model")
    parser.add_argument('--patch_size', type=int, default=16, help="Patch size used in ViT")
    parser.add_argument('--num_transformer_layers', type=int, default=12, help="Number of transformer layers in ViT")
    parser.add_argument('--data_path', type=str, default="/home/groups/comp3710/ADNI/AD_NC", help="Path to the dataset")
    parser.add_argument('--run', type=str, default="brain-viz", help="Path to the dataset")
    
    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    model = ViT(img_size=args.image_size, patch_size=args.patch_size, num_classes=2, num_transformer_layers=args.num_transformer_layers)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Loaded model from {args.model_path} with validation accuracy {checkpoint['val_accuracy']:.2f}%")

    # Set up the data transforms (same as used in training)
    transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.CenterCrop(args.image_size // 1.2),  # Center crop
            transforms.RandomHorizontalFlip(),  # Horizontal flip for augmentation
            transforms.RandomRotation(degrees=10),  # Slight random rotation
            transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),  # Randomly crop and resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1156, 0.1156, 0.1156], std=[0.2229, 0.2229, 0.2229]) 
        ])

    # Load the dataset for prediction (using the test set)
        # Get all the loaders 
    _, _, test_loader = get_dataloaders(batch_size=args.batch_size, image_size=args.image_size, path = args.data_path, shuffle_test=True)
    class_to_idx = test_loader.dataset.class_to_idx

    # Invert the dictionary to get idx-to-class mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print("YOOOO,", args.run)
    if args.run == "predict":
        print("Evaluating model performance...")
        class_report = evaluate_model(model, test_loader, device, idx_to_class=idx_to_class)
        test_loss, test_acc = evaluate(model, test_loader, CrossEntropyLoss(), device)
        print(f"Test Accuracy: {test_acc:.2f}%, and {test_loss=}")
        print("Class report: ", class_report)
        print("Done model performance!")
    elif args.run == "brain-viz":
        # Make predictions and visualize results
        print("Running graph viz!")
        predict_batch(model, test_loader, device, num_images=args.batch_size, idx_to_class = idx_to_class)
    
    # # Generate and print the classification report
    # class_report = evaluate_model(model, test_loader, device)
    # print("Classification Report:")
    # print(class_report)


if __name__ == "__main__":
    main()
