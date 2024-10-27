"""
This code performs inference on the trained models to classify some sample images.

Made by Joshua Deadman
"""

import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch

from modules import SiameseNetwork, BinaryClassifier
from dataset import get_dataloaders

parser = argparse.ArgumentParser()
parser.add_argument("siamese", help="Path to siamese network to be used for inference")
parser.add_argument("classifier", help="Path to classifier to be used for inference")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("WARNING: Using the CPU...")

_, _, sample_loader = get_dataloaders()
print(f"Testing on a sample of {len(sample_loader.dataset)} images.")


# Initialise models
siamese = SiameseNetwork().to(device)
classifier = BinaryClassifier().to(device)
siamese.load_state_dict(torch.load(args.siamese, weights_only=True, map_location=device))
classifier.load_state_dict(torch.load(args.classifier, weights_only=True, map_location=device))

siamese.eval()
classifier.eval()

# Collect feature vectors
sample_features = []
sample_labels = []
with torch.no_grad():
    for anchor, _, _, label in sample_loader:
        features = siamese.forward_once(anchor.to(device))
        sample_features.append(features)
        sample_labels.append(label.to(device))

# Classify the images
with torch.no_grad():
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    for features, labels in zip(sample_features, sample_labels):
        out = classifier(features)
        predicted = torch.argmax(out, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_labels.extend(labels.data.cpu())
        all_predictions.extend(predicted.cpu())

    print('\nTest Accuracy for Classification: {} %'.format(100 * correct / total))

    # Generate a confusion matrix to visualise accuracy
    cm_display = ConfusionMatrixDisplay.from_predictions(all_labels, all_predictions, display_labels=["Benign", "Malignant"])
    cm_display.plot(colorbar=False)
    plt.close()
    plt.show()
