"""
- Predicts melanoma classifications for images in a directory
- Provides evaluation metrics for batch predictions

@author: richardchantra
@student_number: 43032053
"""

import torch
from modules import SiameseNetwork, MLPClassifier, Predict
import argparse

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Predicting melanomas using trained models on a directory of melanoma images")
    parser.add_argument('--folder_path', type=str, default='archive/test-image/image/',
                        help='Directory with new images for prediction')
    parser.add_argument('--siamese_model_path', type=str, default='best_siamese_network.pth',
                        help='Path to the saved Siamese Network model weights')
    parser.add_argument('--mlp_model_path', type=str, default='best_mlp_classifier.pth',
                        help='Path to the saved MLP Classifier model weights')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trained models
    siamese_network = SiameseNetwork().to(device)
    mlp_classifier = MLPClassifier().to(device)
    
    # Load the saved model weights
    siamese_network_checkpoint = torch.load(args.siamese_model_path)
    mlp_classifier_checkpoint = torch.load(args.mlp_model_path)
    siamese_network.load_state_dict(siamese_network_checkpoint['model_state_dict'])
    mlp_classifier.load_state_dict(mlp_classifier_checkpoint['model_state_dict'])
    
    # Create Predict instance
    predictor = Predict(siamese_network, mlp_classifier, device)

    # Run predictions on the specified directory
    predictions, probabilities, image_names = predictor.batch_predict(args.folder_path)
    
    # Evaluate and display results
    results = predictor.evaluate_predictions(predictions, probabilities)
    print(f"\nEvaluation Results for Directory: {args.folder_path}")
    print(f"Benign Count: {results['benign_count']}")
    print(f"Malignant Count: {results['malignant_count']}")
    print(f"Average Probability of Malignant Melanoma: {results['avg_probability']:.2f}")
    print("\nClassification Report:\n", results['classification_report'])

if __name__ == "__main__":
    main()
