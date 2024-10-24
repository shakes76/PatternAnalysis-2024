import torch
import pickle
import argparse
import torchvision.transforms as transforms
from PIL import Image
from modules import SiameseNetwork, Classifier

# Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")


# Argument parser to handle input image path
def parse_arguments():
    parser = argparse.ArgumentParser(description="Classify an input image using the trained Siamese Network and classifier.")
    parser.add_argument('image_path', type=str, help="Path to the input image to be classified.")
    return parser.parse_args()


def load_model():
    # Load the trained Siamese Network model
    model = SiameseNetwork().to(DEVICE)
    model.eval()  # Set model to evaluation mode
    return model


def load_classifier():
    # Load the saved reference embeddings and labels from training
    with open("train_embeddings.pkl", "rb") as f:
        train_embeddings, train_labels = pickle.load(f)

    # Initialize the classifier with the reference set
    classifier = Classifier(margin=0.5)
    classifier.set_reference_set(torch.tensor(train_embeddings), torch.tensor(train_labels))
    return classifier


def preprocess_image(image_path):
    # Load the image and preprocess it
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(DEVICE)


def classify_image(model, classifier, image_tensor):
    # Get embedding for the input image using the Siamese Network
    with torch.no_grad():
        embedding = model(image_tensor).cpu()

    # Predict the class using the classifier
    predicted_class = classifier.predict_class(embedding.squeeze(0))
    return predicted_class


def main():
    # Parse arguments
    args = parse_arguments()

    # Load model and classifier
    model = load_model()
    classifier = load_classifier()

    # Preprocess the input image
    image_tensor = preprocess_image(args.image_path)

    # Classify the image
    predicted_class = classify_image(model, classifier, image_tensor)

    # Print the predicted class
    print(f"Predicted Class: {predicted_class}")


if __name__ == "__main__":
    main()
