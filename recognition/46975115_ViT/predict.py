import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import ADNIDataset
from modules import VisionTransformer

def load_model(model_path, device):
    """
    Loads a pretrained Vision Transformer model from the specified path.

    Args:
        model_path (str): Path to the saved model checkpoint.
        device (torch.device): Device on which to load the model (CPU or GPU).

    Returns:
        VisionTransformer: The loaded and prepared Vision Transformer model.
    """
    
    # Initialize model with weights 
    model = VisionTransformer()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  
    return model

def predict_and_visualize(model, test_loader, device, num_images=16):
    """
    Predicts labels for test images and visualizes the predictions along with true labels.

    Args:
        model (VisionTransformer): The Vision Transformer model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device on which computations are performed.
        num_images (int): Number of images to visualize. Default is 16.
    """
        
    model.eval()
    images_shown = 0 # Counter for displayed images

    # Create a 4x4 grid for visualizing the images
    fig, axs = plt.subplots(4, 4, figsize=(15, 15)) 
    
    # Disable gradient calculations
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) # Forward pass through the model
            _, preds = torch.max(outputs, 1) # Get predicted class indices
            
            # Iterate over batch
            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break
                
                # Convert the tensor back to image
                img = transforms.ToPILImage()(images[i].cpu())

                # Determine the true and predicted labels
                true_label = "AD" if labels[i].item() == 1 else "NC"
                predicted_label = "AD" if preds[i].item() == 1 else "NC"
                
                # Display the image and the labels
                ax = axs[images_shown // 4, images_shown % 4] 
                ax.imshow(img, cmap="gray")
                ax.set_title(f"True: {true_label}, Pred: {predicted_label}")
                ax.axis('off')
                
                # Increment counter
                images_shown += 1
            
            if images_shown >= num_images:
                break
    
    plt.tight_layout()
    plt.savefig("predicted_labels.png")

def main():
    """
    Main function to load the model, prepare the test data, and visualize predictions.
    """
    # Define paths and device configuration
    data_dir = "/home/groups/comp3710/ADNI/AD_NC"
    model_path = "best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the pretrained Vision Transformer model
    model = load_model(model_path, device)
    
    # Define test transforms
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Create the test dataset and DataLoader
    test_dataset = ADNIDataset(root_dir=data_dir, split='test', transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    predict_and_visualize(model, test_loader, device, num_images=16)

if __name__ == "__main__":
    main()

