"""
This script is used for evaluating the GFNet model on a test dataset, specifically for binary classification
tasks (AD vs NC). It loads a trained model, builds the test dataset, and computes the overall accuracy 
and loss on the test set.

Additionally, the script includes a feature for randomly selecting an image from the test set, making a 
prediction, and saving the image with a title displaying the true label and the model's prediction.

@brief: Evaluation script for the GFNet model with accuracy computation and image inference.
@author: Sean Bourchier
"""

import os
import argparse
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from functools import partial
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import build_dataset
from modules import GFNet
from dataset import ADNI_DEFAULT_MEAN_TEST, ADNI_DEFAULT_STD_TEST

def get_args_parser():
    """
    Creates an argument parser for the prediction script.

    Returns:
        argparse.ArgumentParser: Argument parser for command-line options.
    """
    parser = argparse.ArgumentParser('GFNet testing script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int, help='Batch size for testing.')
    parser.add_argument('--arch', default='gfnet-s', type=str, help='Model architecture to use.',
                        choices=['gfnet-ti', 'gfnet-xs', 'gfnet-s', 'gfnet-b']) 
                        # Make sure to change 'default=' to the same model as checkpoint best!
    parser.add_argument('--input-size', default=224, type=int, help='Input image size.')
    parser.add_argument('--data-path', default='data/', type=str, help='Path to the dataset.')
    parser.add_argument('--data-set', default='ADNI', type=str, help='Dataset name.')
    parser.add_argument('--seed', default=1, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--model-path', default='outputs/checkpoint_best.pth', help='Path to the model checkpoint.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of workers for data loading.')
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for better GPU transfer.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='Do not pin memory.')
    parser.set_defaults(pin_mem=True)
    return parser

def main(args):
    """
    Main function for testing the GFNet model.
        - Validate model on test set
        - Return accuracy and produce Confusion Matrix
        - Run inference on single random image from test set

    Args:
        args: Parsed command-line arguments.
    """
    # Set the device (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cudnn.benchmark = True
        print("CUDA is available. Using GPU with CUDA.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available. Using GPU with Metal (M1/M2).")
    else:
        device = torch.device("cpu")
        print("Neither CUDA nor MPS are available. Using CPU.")

    # Build the test dataset and DataLoader
    dataset_test, _ = build_dataset(split='test', args=args)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Create and configure the model
    model = create_model(args)
    model.default_cfg = _cfg()
    model = load_model_weights(model, args.model_path)
    model = model.to(device)
    print(f'## Model loaded with {sum(p.numel() for p in model.parameters())} parameters.')

    # Set the criterion (loss function)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Validate the model on the test data
    print('## Validating model on test dataset...')
    # validate(data_loader_test, model, criterion, device)
    validate(data_loader_test, model, criterion, device, class_names=('NC', 'AD'), output_dir='outputs')
    # Making a single inference
    print('## Making a single inference and saving to output.')
    save_random_test_image(data_loader_test, model, device, class_names=('NC', 'AD'), output_dir='outputs')


def create_model(args):
    """
    Creates an instance of the GFNet model based on architecture.

    Args:
        args: Parsed command-line arguments.

    Returns:
        GFNet: The instantiated model.
    """
    print(f"Creating model: {args.arch}")
    if args.arch == 'gfnet-xs':
        return GFNet(img_size=args.input_size, patch_size=16, embed_dim=384, depth=12, mlp_ratio=4,
                     norm_layer=partial(nn.LayerNorm, eps=1e-6))
    elif args.arch == 'gfnet-ti':
        return GFNet(img_size=args.input_size, patch_size=16, embed_dim=256, depth=12, mlp_ratio=4,
                     norm_layer=partial(nn.LayerNorm, eps=1e-6))
    elif args.arch == 'gfnet-s':
        return GFNet(img_size=args.input_size, patch_size=16, embed_dim=384, depth=19, mlp_ratio=4,
                     norm_layer=partial(nn.LayerNorm, eps=1e-6))
    elif args.arch == 'gfnet-b':
        return GFNet(img_size=args.input_size, patch_size=16, embed_dim=512, depth=19, mlp_ratio=4,
                     norm_layer=partial(nn.LayerNorm, eps=1e-6))
    else:
        raise NotImplementedError(f"Architecture '{args.arch}' is not implemented.")


def load_model_weights(model, model_path):
    """
    Loads pretrained weights into the model.

    Args:
        model (nn.Module): The model instance.
        model_path (str): Path to the model checkpoint.

    Returns:
        nn.Module: Model with loaded weights.
    """
    print(f"Loading model weights from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    print('Model weights loaded successfully.')
    return model


def accuracy(output, target):
    """
    Computes the accuracy for binary classification.
    
    Args:
        output (torch.Tensor): Model output logits of shape (batch_size, num_classes).
        target (torch.Tensor): Ground truth labels of shape (batch_size).
        
    Returns:
        float: Accuracy as a percentage.
    """
    with torch.no_grad():
        # Get the predicted class (0 or 1) by taking the index of the maximum logit.
        pred = output.argmax(dim=1)
        correct = (pred == target).sum().item()
        accuracy = correct / target.size(0) * 100
        return accuracy

def validate(val_loader, model, criterion, device, class_names=('NC', 'AD'), output_dir='outputs'):
    """
    Evaluates the model on the entire validation dataset for binary classification
    and saves a confusion matrix as an image.

    Args:
        val_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): Model to evaluate.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the model on.
        output_dir (str): Directory to save the confusion matrix image.
        class_names (tuple): Tuple containing the class names.
    
    Returns:
        float: Overall accuracy of the model on the validation dataset.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Compute output and loss
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)

            # Compute the predictions
            _, predicted = outputs.max(1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            # Store all targets and predictions for the confusion matrix
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate average loss and accuracy
    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    print(f'Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Create and save the confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    os.makedirs(output_dir, exist_ok=True)
    cm_output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.title('Confusion Matrix')
    plt.savefig(cm_output_path)
    plt.close()
    print(f'Confusion matrix saved to {cm_output_path}')
    return accuracy


def save_random_test_image(val_loader, model, device, class_names=('NC', 'AD'), output_dir='outputs'):
    """
    Randomly selects an image from the test dataset, makes a prediction, and saves the image
    with a title showing the true and predicted labels.

    Args:
        val_loader (DataLoader): DataLoader for validation/test data.
        model (nn.Module): The trained model.
        device (torch.device): Device to run the model on.
        class_names (tuple): Tuple containing the class names (default: ('NC', 'AD')).
        output_dir (str): Directory to save the output image (default: 'output_images').
    """
    model.eval()
    # Select a random batch and a random image from the batch
    images, targets = next(iter(val_loader))
    random_idx = random.randint(0, len(images) - 1)
    image = images[random_idx]
    target = targets[random_idx]
    image = image.unsqueeze(0).to(device)
    target = target.item()
    # Make a prediction using the model
    with torch.no_grad():
        output = model(image)
        predicted = output.argmax(dim=1).item()
    # Convert the image tensor to a NumPy array for plotting
    image_np = image.squeeze().cpu().numpy()
    if image_np.ndim == 3 and image_np.shape[0] == 1:
        image_np = image_np[0]
    # Get the true and predicted class names
    true_label = class_names[target]
    predicted_label = class_names[predicted]
    os.makedirs(output_dir, exist_ok=True)
    # Plot the image with the true and predicted labels as the title
    plt.figure()
    plt.imshow(image_np, cmap='gray')
    plt.title(f'True Label = {true_label}, Predicted = {predicted_label}')
    plt.axis('off')
    # Save image
    output_path = os.path.join(output_dir, f'random_test_image_{random_idx}.png')
    plt.savefig(output_path)
    plt.close()
    print(f'Image saved to {output_path} with True Label = {true_label} and Predicted = {predicted_label}')

def _cfg(url='', **kwargs):
    """
    Helper function to create model configuration.
    
    Args:
        url (str): URL for pretrained model weights.
        **kwargs: Additional configuration parameters.
        
    Returns:
        dict: Configuration dictionary for model initialization.
    """
    return {
        'url': url,
        'num_classes': 2, 
        'input_size': (1, 224, 224), 
        'pool_size': None,
        'crop_pct': 0.9, 
        'interpolation': 'bicubic',
        'mean': ADNI_DEFAULT_MEAN_TEST, 
        'std': ADNI_DEFAULT_STD_TEST,
        'first_conv': 'patch_embed.proj', 
        'classifier': 'head',
        **kwargs
    }

if __name__ == '__main__':
    # Parse command-line arguments and start the main function
    parser = argparse.ArgumentParser('GFNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
