import torch
from modules import SiameseNetwork, MLPClassifier, Predict

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trained models
    siamese_network = SiameseNetwork().to(device)
    mlp_classifier = MLPClassifier().to(device)
    
    # Load the saved model weights
    siamese_network_checkpoint = torch.load('best_siamese_network.pth')
    mlp_classifier_checkpoint = torch.load('best_mlp_classifier.pth')
    siamese_network.load_state_dict(siamese_network_checkpoint['model_state_dict'])
    mlp_classifier.load_state_dict(mlp_classifier_checkpoint['model_state_dict'])
    
    # Create Predict instance
    predictor = Predict(siamese_network, mlp_classifier, device)
    
    # Path to the folder with new images for prediction
    folder_path = 'archive/test-image/image/'  # Replace with your folder path

    # Run predictions on the folder
    predictions, probabilities, image_names = predictor.batch_predict(folder_path)
    
    # Evaluate and display results
    results = predictor.evaluate_predictions(predictions, probabilities)
    print(f"\nEvaluation Results: {folder_path}")
    print(f"Benign Count: {results['benign_count']}")
    print(f"Malignant Count: {results['malignant_count']}")
    print(f"Average Probability of Malignant Melanoma: {results['avg_probability']:.2f}")
    print("\nClassification Report:\n", results['classification_report'])

if __name__ == "__main__":
    main()