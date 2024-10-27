"""
- Trains a Siamese Network and MLP Classifier on melanoma images
- Evaluates model performance after training

@author: richardchantra
@student_number: 43032053
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import DataManager
from modules import SiameseNetwork, MLPClassifier, Evaluate, Predict

def train_siamese_network(siamese_network, optimizer, train_loader, epochs=5, margin=1.0):
    """
    Train Siamese Network to learn embeddings from images
    """
    siamese_network.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Siamese"):
            # Get batch data
            img1 = batch['img1'].to(device)
            img2 = batch['img2'].to(device)
            similarity_label = batch['similarity_label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            embedding1, embedding2 = siamese_network(img1, img2)
            
            # Calculate loss
            loss = siamese_network.contrastive_loss(embedding1, embedding2, similarity_label, margin)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(siamese_network.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Siamese Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': siamese_network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_siamese_network.pth')

def train_mlp_classifier(siamese_network, mlp_classifier, optimizer, train_loader, epochs=5):
    """
    Train MLP classifier using Siamese Network embeddings
    """
    mlp_classifier.train()
    siamese_network.eval() # Freeze Siamese network
    criterion = nn.BCELoss()
    best_acc = 0.0
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Classifier"):
            # Get image and its label
            img1 = batch['img1'].to(device)
            diagnosis_label = batch['diagnosis1'].to(device).unsqueeze(1)
            
            # Get embeddings without gradient tracking
            with torch.no_grad():
                embeddings = siamese_network.get_embedding(img1)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = mlp_classifier(embeddings)
            loss = criterion(outputs, diagnosis_label)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            total += diagnosis_label.size(0)
            correct += (predicted == diagnosis_label).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Classifier Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': mlp_classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, 'best_mlp_classifier.pth')


def main():
    # Set device
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup data
    data_manager = DataManager('archive/train-metadata.csv', 'archive/train-image/image/')
    data_manager.load_data()
    data_manager.create_dataloaders()
    train_loader = data_manager.train_loader
    test_loader = data_manager.test_loader

    # Initialize models
    siamese_network = SiameseNetwork().to(device)
    mlp_classifier = MLPClassifier().to(device)
    
    # Initialize optimizers
    optimizer_siamese = optim.Adam(siamese_network.parameters(), lr=0.001, weight_decay=5e-5)
    optimizer_mlp = optim.Adam(mlp_classifier.parameters(), lr=0.001, weight_decay=1e-4)

    print("Training Siamese Network to learn embeddings from images:")
    train_siamese_network(siamese_network, optimizer_siamese, train_loader, epochs=16)
    
   # Train the MLP classifier
    print("\nTraining MLPClassifier using learned embeddings:")
    train_mlp_classifier(siamese_network, mlp_classifier, optimizer_mlp, train_loader, epochs=8)

    # Evaluate the trained model
    print("\nEvaluating the model on test data:")
    predictor = Predict(siamese_network, mlp_classifier, device)
    preds, probs, labels = predictor.predict(test_loader)

    evaluator = Evaluate(preds, probs, labels)
    results = evaluator.evaluate()

    print("\nEvaluation Results:\n")
    print(f"Overall Accuracy: {results['basic_metrics']['accuracy']}")
    print(f"Malignant Accuracy: {results['basic_metrics']['malignant_accuracy']}")
    print(f"ROC-AUC Score: {results['roc_auc']['auc']}\n")
    print(results['class_report'])
    
    # Optionally save and plot results
    evaluator.plot_results()
    evaluator.save_results()

if __name__ == "__main__":
    main()