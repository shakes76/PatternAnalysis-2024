import torch
from dataset import DataManager
from modules import SiameseNetwork, MLPClassifier, Predict, Evaluate

def main():
        # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup data
    data_manager = DataManager('archive/train-metadata.csv', 'archive/train-image/image/')
    data_manager.load_data()
    data_manager.create_dataloaders()
    test_loader = data_manager.test_loader
    
    # Load models
    siamese_network = SiameseNetwork().to(device)
    mlp_classifier = MLPClassifier().to(device)
    
    # Load saved weights from training
    siamese_network_checkpoint = torch.load('best_siamese_network.pth')
    mlp_classifier_checkpoint = torch.load('best_mlp_classifier.pth')
    siamese_network.load_state_dict(siamese_network_checkpoint['model_state_dict'])
    mlp_classifier.load_state_dict(mlp_classifier_checkpoint['model_state_dict'])
    
    # Create Predict instance and run predictions
    predictor = Predict(siamese_network, mlp_classifier, device)
    preds, probs, labels = predictor.predict(test_loader)
    
    # Create Evaluate instance and run evaluation
    evaluator = Evaluate(preds, probs, labels)
    results = evaluator.evaluate()
    
    # Print evaluation results
    print("Evaluation\n")
    print(f"Overall Accuracy: {results['basic_metrics']['accuracy']}")
    print(f"Malignant Accuracy: {results['basic_metrics']['malignant_accuracy']}")
    print(f"ROC-AUC Score: {results['roc_auc']['auc']}\n")
    print(results['class_report'])
    
    # Generate and save plots
    evaluator.plot_results()
    evaluator.save_results()


if __name__ == "__main__":
    main()