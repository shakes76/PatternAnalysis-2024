import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from modules import ViT 
from pathlib import Path
import argparse
import torch.nn.functional as F  # Import this for applying softmax


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

def predict_batch(model, dataloader, device, num_images=32, font_size=8):
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
            imshow(images[i].cpu(), label=labels[i].item(), prediction=(preds[i].item(), prob), font_size=font_size)
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser()

    # Add argument for model path
    parser.add_argument('--model_path', type=str, default="models/best_model.pth", help="Path to the trained model")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for prediction")
    parser.add_argument('--image_size', type=int, default=224, help="Image size for the model")
    parser.add_argument('--patch_size', type=int, default=16, help="Patch size used in ViT")
    parser.add_argument('--num_transformer_layers', type=int, default=12, help="Number of transformer layers in ViT")
    parser.add_argument('--data_path', type=str, default="/home/groups/comp3710/ADNI/AD_NC", help="Path to the dataset")
    
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
    val_dir = f"{args.data_path}/test"
    val_data = datasets.ImageFolder(root=val_dir, transform=transform)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

    # Make predictions and visualize results
    predict_batch(model, val_loader, device, num_images=args.batch_size)


if __name__ == "__main__":
    main()
