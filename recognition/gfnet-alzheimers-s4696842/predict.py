import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from .modules import GFNet
from .utils import ADNI_CLASSES, get_device
from .dataset import ADNI_IMAGE_DIMENSIONS


def load_model(model_path, device, num_classes=2):
    """
    Loads the GFNet model from a saved state dictionary.

    Args:
        model_path (str): Path to the saved model file.
        device (torch.device): The device to load the model onto.
        num_classes (int, optional): Number of classes for classification. Defaults to 2.

    Returns:
        nn.Module: The loaded GFNet model.
    """
    model = GFNet(
        img_size=ADNI_IMAGE_DIMENSIONS,
        in_chans=1,
        num_classes=num_classes,
        depth=12,
        embed_dim=260,
        drop_rate=0.1,
        drop_path_rate=0.1,
        patch_size=(8, 8),
    )
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    """
    Preprocesses the input image for prediction.

    Args:
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(ADNI_IMAGE_DIMENSIONS),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def predict(model, image_tensor, device):
    """
    Performs prediction on a preprocessed image tensor.

    Args:
        model (nn.Module): The loaded GFNet model.
        image_tensor (torch.Tensor): The preprocessed image tensor.
        device (torch.device): The device to perform computation on.

    Returns:
        int: The predicted class index.
        torch.Tensor: The softmax probabilities for each class.
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = nn.functional.softmax(outputs, dim=1)
        predicted_class = probabilities.argmax(dim=1).item()
    return predicted_class, probabilities.cpu()


def parse_args():
    """
    Parses command-line arguments for the prediction script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="GFNet Image Prediction Script")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the saved model file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device_str = get_device()
    device = torch.device(device_str)
    print(f"Using device: {device_str}")

    model = load_model(args.model, device, num_classes=len(ADNI_CLASSES))
    image_tensor = preprocess_image(args.image)
    print(f"Image tensor shape: {image_tensor.shape}")

    predicted_class_idx, probabilities = predict(model, image_tensor, device)
    predicted_class_name = ADNI_CLASSES[predicted_class_idx]
    print(f"Predicted Class: {predicted_class_name}")
    print(f"Probabilities: {probabilities.numpy()}")

    for idx, class_name in enumerate(ADNI_CLASSES):
        prob = probabilities[0][idx].item()
        print(f"Probability of {class_name}: {prob:.4f}")
