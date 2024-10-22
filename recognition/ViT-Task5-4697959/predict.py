# predict.py

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from modules import VisionTransformer
import os


def load_model(model_path, device, img_size=224, patch_size=16, emb_size=768, num_heads=12, depth=12, ff_dim=3072, num_classes=2, dropout=0.1, cls_token=True):
    """
    Loads the trained Vision Transformer model.

    Args:
        model_path (str): Path to the saved model weights.
        device (torch.device): Device to load the model on.
        Other args: Model architecture parameters.

    Returns:
        nn.Module: Loaded model.
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        emb_size=emb_size,
        num_heads=num_heads,
        depth=depth,
        ff_dim=ff_dim,
        num_classes=num_classes,
        dropout=dropout,
        cls_token=cls_token
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path, img_size=224):
    """
    Preprocesses the input image.

    Args:
        image_path (str): Path to the image.
        img_size (int): Size to resize the image.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(model, image_tensor, device):
    """
    Predicts the class of the input image.

    Args:
        model (nn.Module): Trained model.
        image_tensor (torch.Tensor): Preprocessed image tensor.
        device (torch.device): Device to perform inference on.

    Returns:
        int: Predicted class index.
        torch.Tensor: Output logits.
    """
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
    return preds.item(), outputs

def visualize_prediction(image_path, predicted_class, probabilities, class_names, save_dir='saved_models'):
    """
    Visualizes the image with prediction probabilities.

    Args:
        image_path (str): Path to the image.
        predicted_class (str): Predicted class name.
        probabilities (numpy.ndarray): Probabilities for each class.
        class_names (list): List of class names.
        save_dir (str): Directory to save the visualization.
    """
    image = Image.open(image_path).convert('RGB')

    plt.figure(figsize=(8,6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class}\nProbabilities: {probabilities}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'prediction_visualization.png'))
    plt.show()

def main():
    # Configuration
    model_path = "saved_models/best_vit_model.pth"  # Path to your trained model
    image_path = "/home/groups/comp3710/ADNI/AD_NC/test/AD/1003730_100.jpeg"  # Replace with an actual image path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 224
    patch_size = 16
    emb_size = 768
    num_heads = 12
    depth = 12
    ff_dim = 3072
    num_classes = 2
    dropout = 0.1 # change dropout here 
    cls_token = True
    class_names = ['AD', 'NC'] 

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}. Please provide a valid path.")
        return

    # Load the model
    model = load_model(
        model_path=model_path,
        device=device,
        img_size=img_size,
        patch_size=patch_size,
        emb_size=emb_size,
        num_heads=num_heads,
        depth=depth,
        ff_dim=ff_dim,
        num_classes=num_classes,
        dropout=dropout,
        cls_token=cls_token
    )
    print("Model loaded successfully.")

    # Preprocess the image
    image_tensor = preprocess_image(image_path, img_size=img_size)

    # Make prediction
    pred_idx, outputs = predict(model, image_tensor, device)
    predicted_class = class_names[pred_idx]
    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    print(f"Prediction for the image '{os.path.basename(image_path)}': {predicted_class}")
    print(f"Probabilities: {probabilities}")

    # Visualize prediction
    visualize_prediction(image_path, predicted_class, probabilities, class_names, save_dir='saved_models')

if __name__ == '__main__':
    main()








