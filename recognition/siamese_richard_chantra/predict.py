import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from train import SiameseNetwork, MLPClassifier
from dataset import test_loader

# Precision set to 3dp
np.set_printoptions(precision=3, suppress=True)

class Predict:
    def __init__(self, siamese_network, mlp_classifier, device):
        self.siamese_network = siamese_network
        self.mlp_classifier = mlp_classifier
        self.device = device

    def predict(self, data_loader):
        """Run predictions on the given data loader"""
        # Set models to evaluation mode
        self.siamese_network.eval()
        self.mlp_classifier.eval()
        
        preds = []
        probs = []
        labels = []
        
        with torch.no_grad():  # No need to track gradients for prediction
            for batch in tqdm(data_loader, desc="Predicting"):
                # Move batch to GPU if available
                images = batch['img1'].to(self.device) # Get first image from pair
                batch_labels = batch['diagnosis1'].to(self.device) # Get true label
                
                # Get embeddings from Siamese Network
                embeddings = self.siamese_network.get_embedding(images)
                
                # Probability of being malignant
                batch_probs = self.mlp_classifier(embeddings).squeeze() 
                batch_preds = (batch_probs > 0.5).float()
                
                # Store results using CPU
                preds.extend(batch_preds.cpu().numpy())
                probs.extend(batch_probs.cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
        
        # Convert lists to numpy arrays
        return np.array(preds), np.array(probs), np.array(labels)

class Evaluate:
    def __init__(self, preds, probs, labels):
        self.preds = preds
        self.probs = probs
        self.labels = labels

    def evaluate(self):
        """Evaluate predictions and return metrics"""
        return {
            'basic_metrics': self._get_basic_metrics(),
            'roc_auc': self._get_roc_auc(),
            'class_report': classification_report(self.labels, self.preds, 
                                               target_names=['Benign', 'Malignant'])
        }

    def _get_basic_metrics(self):
        """Calculate accuracy metrics for both classes"""
        accuracy = (self.preds == self.labels).mean()
        malignant_mask = self.labels == 1
        benign_mask = self.labels == 0
        
        return {
            'accuracy': accuracy,
            'malignant_accuracy': (self.preds[malignant_mask] == self.labels[malignant_mask]).mean(),
            'benign_accuracy': (self.preds[benign_mask] == self.labels[benign_mask]).mean()
        }
    
    def _get_roc_auc(self):
        """Calculate ROC curve and AUC score"""
        fpr, tpr, _ = roc_curve(self.labels, self.probs)
        return {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}

    def plot_results(self):
        """Generate ROC curve and confusion matrix plots"""
        # Plot ROC curve
        roc_data = self._get_roc_auc()
        plt.figure(figsize=(10, 8))
        sns.lineplot(x=roc_data['fpr'], y=roc_data['tpr'])
        sns.lineplot(x=[0, 1], y=[0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (AUC = {roc_data["auc"]:.3f})')
        plt.savefig('roc_curve.png')
        plt.close()
        
        # Plot confusion matrix
        cm = confusion_matrix(self.labels, self.preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='vlag',
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'])
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()

    def save_results(self, filename='evaluation_results.txt'):
        """Save all evaluation metrics to file"""
        results = self.evaluate()
        metrics = results['basic_metrics']
        
        with open(filename, 'w') as f:
            f.write("Evaluation Results\n\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']}\n")
            f.write(f"Malignant Accuracy: {metrics['malignant_accuracy']}\n")
            f.write(f"Benign Accuracy: {metrics['benign_accuracy']}\n")
            f.write(f"ROC-AUC Score: {results['roc_auc']['auc']}\n\n")
            f.write(f"Classification Report:\n{results['class_report']}\n")

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    siamese_network = SiameseNetwork().to(device)
    mlp_classifier = MLPClassifier().to(device)
    
    # Load saved weights from training
    siamese_network_checkpoint = torch.load('best_siamese_network.pth')
    mlp_classifier_checkpoint = torch.load('best_mlp_classifier_model.pth')
    siamese_network.load_state_dict(siamese_network_checkpoint['model_state_dict'])
    mlp_classifier.load_state_dict(mlp_classifier_checkpoint['model_state_dict'])
    
    # Create Predict instance and run predictions
    predictor = Predict(siamese_network, mlp_classifier, device)
    preds, probs, labels = predictor.predict(test_loader)
    
    # Create Evaluate instance and run evaluation
    evaluator = Evaluate(preds, probs, labels)
    results = evaluator.evaluate()
    
    # Print evaluation results
    print("Evaluation\n\n")
    print(f"Overall Accuracy: {results['basic_metrics']['accuracy']}")
    print(f"Malignant Accuracy: {results['basic_metrics']['malignant_accuracy']}")
    print(f"ROC-AUC Score: {results['roc_auc']['auc']}\n")
    print(results['class_report'])
    
    # Generate and save plots
    evaluator.plot_results()
    evaluator.save_results()
