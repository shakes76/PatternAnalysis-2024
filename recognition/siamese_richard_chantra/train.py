"""
- Trains a Siamese Network and MLP Classifier on melanoma images
- Evaluates model performance after training

@author: richardchantra
@student_number: 43032053
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import DataManager
from modules import SiameseNetwork, MLPClassifier, Evaluate, Predict
import argparse

def train_siamese_network(siamese_network, optimizer, train_loader, epochs=5, margin=1.0):
    """
    Train Siamese Network to learn embeddings from images
    """
    siamese_network.train()
    best_loss = float('inf')
    epoch_losses = []
    
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
            torch.nn.utils.clip_grad_norm_(siamese_network.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
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
    return epoch_losses

def train_mlp_classifier(siamese_network, mlp_classifier, optimizer, train_loader, epochs=5):
    """
    Train MLP classifier using Siamese Network embeddings
    """
    mlp_classifier.train()
    siamese_network.eval()
    criterion = nn.BCELoss()
    best_acc = 0.0
    epoch_losses = []
    
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
        epoch_losses.append(epoch_loss)
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
    return epoch_losses

def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Training a Siamese Network and MLP Classifier on melanoma images")
    parser.add_argument('--csv_path', type=str, default='archive/train-metadata.csv',
                        help='Path to the CSV metadata file')
    parser.add_argument('--img_dir', type=str, default='archive/train-image/image/',
                        help='Directory path to the image files')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for DataLoader')
    parser.add_argument('--epochs_siamese', type=int, default=16,
                        help='Number of epochs for training the Siamese Network')
    parser.add_argument('--epochs_mlp', type=int, default=8,
                        help='Number of epochs for training the MLP Classifier')
    parser.add_argument('--save_dir', type=str, default="plots",
                        help='Directory to save training plots')
    args = parser.parse_args()

    # Set device
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup data
    data_manager = DataManager(args.csv_path, args.img_dir)
    data_manager.load_data()
    data_manager.create_dataloaders(batch_size=args.batch_size)
    train_loader = data_manager.train_loader
    test_loader = data_manager.test_loader

    # Initialize models
    siamese_network = SiameseNetwork().to(device)
    mlp_classifier = MLPClassifier().to(device)
    
    # Initialize optimizers
    optimizer_siamese = optim.Adam(
        siamese_network.parameters(),
        lr=0.001,
        weight_decay=5e-5
    )
    optimizer_mlp = optim.Adam(
        mlp_classifier.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    # Train the Siamese Network
    print("Training Siamese Network to learn embeddings from images:")
    siamese_losses = train_siamese_network(
        siamese_network,
        optimizer_siamese,
        train_loader,
        epochs=args.epochs_siamese
    )
    
    # Train the MLP classifier
    print("\nTraining MLPClassifier using learned embeddings:")
    mlp_losses = train_mlp_classifier(
        siamese_network,
        mlp_classifier, 
        optimizer_mlp, 
        train_loader,
        epochs=args.epochs_mlp
    )

    # Plot and save training losses
    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)
    
    Evaluate.plot_loss(
        siamese_losses,
        title="Siamese Network Training Loss per Epoch",
        xlabel="Epoch",
        ylabel="Loss",
        save_path=os.path.join(save_dir, "siamese_network_loss.png"),
        color='b',
        marker='o'
    )
    Evaluate.plot_loss(
        mlp_losses,
        title="MLP Classifier Training Loss per Epoch", 
        xlabel="Epoch", 
        ylabel="Loss", 
        save_path=os.path.join(save_dir, "mlp_classifier_loss.png"),
        color='g', 
        marker='s'
    )

    # Evaluate the model after training
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
    
    # Optionally save and plot evaluation results
    evaluator.plot_results()
    evaluator.save_results()

if __name__ == "__main__":
    main()
