"""
- The Siamese Network, MLP Classifier and Contrastive Loss are defined for use in train.py and predict.py
- The Predict class is defined for prediction using a saved model in predict.py
- The Evaluation class is defined for evaluating the performance post training in train.py

@author: richardchantra
@student_number: 43032053
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import seaborn as sns
from PIL import Image

class SiameseNetwork(nn.Module):
    """
    Siamese Network for learning image embeddings of benign and malignant melanomas.
    """
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # ResNet50 Feature Extractor
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x1, x2):
        """
        Forward pass to compute embeddings for a pair of images
        """
        # Get embeddings for both images
        out1 = self.get_embedding(x1)
        out2 = self.get_embedding(x2)
        return out1, out2
    
    def get_embedding(self, x):
        """
        Computing embeddings for a single image
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)
    
    def contrastive_loss(self, output1, output2, label, margin=1.0):
        """
        Contrastive loss for Siamese Network training
        """
        # Calculate euclidean distance
        euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1) + 1e-6)
        
        # Calculate contrastive loss
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                        label * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
        
        return loss

class MLPClassifier(nn.Module):
    """
    MLP Classifier using Siamese Network embeddings to predict melanoma
    """
    def __init__(self, embedding_dim=128):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, embedding):
        """
        Input: embedding from Siamese network
        Output: probability of being malignant (0 = benign, 1 = malignant)
        """
        return self.classifier(embedding)



class Predict:
    """
    Handling prediction for images and using a trained SiameseNetwork and MLPClassifier to do so
    """
    def __init__(self, siamese_network, mlp_classifier, device):
        self.siamese_network = siamese_network
        self.mlp_classifier = mlp_classifier
        self.device = device

    @staticmethod
    def load_image(image_path):
        """
        Load and preprocess an image for prediction.
        """
        # Apply transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)
    
    def predict_image(self, image_path):
        """
        Predict whether a new image is benign or malignant.
        """
        # Load and preprocess the image
        image = self.load_image(image_path).to(self.device)
        
        # Eval mode
        self.siamese_network.eval()
        self.mlp_classifier.eval()
        
        with torch.no_grad():
            # Generate embedding for the image
            embedding = self.siamese_network.get_embedding(image)
            
            # Classify the embedding
            output = self.mlp_classifier(embedding)
            prediction = (output > 0.5).float() 
            probability = output.item()
        
        return prediction.item(), probability

    def batch_predict(self, folder):
        """
        Performs predictions on all images within a specified directory
        """
        predictions = []
        probabilities = []
        image_names = []
        
        for filename in tqdm(os.listdir(folder), desc="Predicting images"):
            if filename.endswith(('.jpg')):
                image_path = os.path.join(folder, filename)
                prediction, probability = self.predict_image(image_path)
                
                predictions.append(prediction)
                probabilities.append(probability)
                image_names.append(filename)
        
        return predictions, probabilities, image_names

    def evaluate_predictions(self, predictions, probabilities):
        """
        Evaluates a set of metrics for a batch of predictions
        """
        benign_count = predictions.count(0)
        malignant_count = predictions.count(1)
        avg_probability = np.mean(probabilities)
        
        report = classification_report(
            predictions, [1]*len(predictions), target_names=['Benign', 'Malignant']
        )

        return {
            'benign_count': benign_count,
            'malignant_count': malignant_count,
            'avg_probability': avg_probability,
            'classification_report': report
        }

    def predict(self, data_loader):
        """
        Run predictions on the given data loader
        """
        # Set models to evaluation mode
        self.siamese_network.eval()
        self.mlp_classifier.eval()
        
        preds = []
        probs = []
        labels = []
        
        with torch.no_grad():
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
    """
    Evaluating the classifier using a number of metrics
    """
    def __init__(self, preds, probs, labels):
        self.preds = preds
        self.probs = probs
        self.labels = labels

    def evaluate(self):
        """
        Evaluate predictions and return metrics
        """
        return {
            'basic_metrics': self._get_basic_metrics(),
            'roc_auc': self._get_roc_auc(),
            'class_report': classification_report(self.labels, self.preds, 
                                               target_names=['Benign', 'Malignant'])
        }

    def _get_basic_metrics(self):
        """
        Calculate accuracy metrics for both classes
        """
        accuracy = (self.preds == self.labels).mean()
        malignant_mask = self.labels == 1
        benign_mask = self.labels == 0
        
        return {
            'accuracy': accuracy,
            'malignant_accuracy': (self.preds[malignant_mask] == self.labels[malignant_mask]).mean(),
            'benign_accuracy': (self.preds[benign_mask] == self.labels[benign_mask]).mean()
        }
    
    def _get_roc_auc(self):
        """
        Calculate ROC curve and AUC score
        """
        fpr, tpr, _ = roc_curve(self.labels, self.probs)
        return {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}

    def plot_results(self):
        """
        Generate ROC curve and confusion matrix plots
        """
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
        """
        Save evaluation metrics to file
        """
        results = self.evaluate()
        metrics = results['basic_metrics']
        
        with open(filename, 'w') as f:
            f.write("\nEvaluation Results:\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']}\n")
            f.write(f"Malignant Accuracy: {metrics['malignant_accuracy']}\n")
            f.write(f"Benign Accuracy: {metrics['benign_accuracy']}\n")
            f.write(f"ROC-AUC Score: {results['roc_auc']['auc']}\n\n")
            f.write(f"Classification Report:\n{results['class_report']}\n")

    @staticmethod
    def plot_loss(self, data, title, xlabel, ylabel, save_path, color='b', marker='o'):
        """
        Plots and saves a training loss graph
        """
        plt.figure(figsize=(8, 6))
        plt.plot(data, label=title, marker=marker, color=color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Plot saved at: {save_path}")
        