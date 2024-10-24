import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import get_data_loaders
from modules import SiameseNetwork, Classifier
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import LpDistance
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Local flag to adjust parameters for testing
LOCAL = True

# Parameters
NUM_EPOCHS = 60 if not LOCAL else 10
LEARNING_RATE = 0.001
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
SUBSET_SIZE = 0.01

def train():
    # Load data loaders without sampler for now
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)

    # If LOCAL, use a subset of the train and validation datasets for faster testing
    if LOCAL:
        train_subset_indices = np.random.choice(len(train_loader.dataset), size=int(SUBSET_SIZE * len(train_loader.dataset)), replace=False)
        train_subset = Subset(train_loader.dataset, train_subset_indices)
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

        val_subset_indices = np.random.choice(len(val_loader.dataset), size=int(SUBSET_SIZE * len(val_loader.dataset)), replace=False)
        val_subset = Subset(val_loader.dataset, val_subset_indices)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    # Load the model from modules.py
    model = SiameseNetwork().to(DEVICE)

    # Define distance metric and contrastive loss function
    distance = LpDistance(normalize_embeddings=False)
    criterion = losses.ContrastiveLoss(pos_margin=0, neg_margin=1, distance=distance)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)

    # Lists to store loss values for plotting
    train_losses = []
    val_losses = []

    # Lists to store embeddings and labels for visualization
    train_embeddings_list = []
    train_labels_list = []
    val_embeddings_list = []
    val_labels_list = []

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()  # Set model to training mode
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            # Unpack the batch
            img, labels = batch

            # Move to device
            img, labels = img.to(DEVICE), labels.to(DEVICE)

            # Forward pass to get the embeddings
            embeddings = model(img)

            # Calculate contrastive loss
            loss = criterion(embeddings, labels)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Step the learning rate scheduler
        scheduler.step()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation Loop
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            val_loss = 0
            for val_batch in tqdm(val_loader, desc="Validation"):
                val_img, val_labels = val_batch
                val_img, val_labels = val_img.to(DEVICE), val_labels.to(DEVICE)

                val_embeddings = model(val_img)

                val_loss += criterion(val_embeddings, val_labels).item()

                # Store validation embeddings and labels for visualization
                val_embeddings_list.append(val_embeddings.cpu())
                val_labels_list.append(val_labels.cpu())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Print summary of training and validation losses for each epoch
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Summary: Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Store training embeddings and labels for the final epoch
        if epoch == NUM_EPOCHS - 1:
            with torch.no_grad():
                for batch in train_loader:
                    img, labels = batch
                    img, labels = img.to(DEVICE), labels.to(DEVICE)
                    embeddings = model(img)
                    train_embeddings_list.append(embeddings.cpu())
                    train_labels_list.append(labels.cpu())

    # Concatenate all embeddings and labels
    train_embeddings = torch.cat(train_embeddings_list, dim=0).numpy()
    train_labels = torch.cat(train_labels_list, dim=0).numpy()
    val_embeddings = torch.cat(val_embeddings_list, dim=0).numpy()
    val_labels = torch.cat(val_labels_list, dim=0).numpy()

    # Save embeddings for later use
    with open("train_embeddings.pkl", "wb") as f:
        pickle.dump((train_embeddings, train_labels), f)

    with open("val_embeddings.pkl", "wb") as f:
        pickle.dump((val_embeddings, val_labels), f)

    # Plotting the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')

    # PCA and t-SNE for Visualization
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)

    # Reduce dimensions using PCA
    train_pca = pca.fit_transform(train_embeddings)
    val_pca = pca.transform(val_embeddings)

    # Apply t-SNE
    train_tsne = tsne.fit_transform(train_pca)
    val_tsne = tsne.fit_transform(val_pca)

    # Plotting the PCA + t-SNE reduced embeddings
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(train_tsne[:, 0], train_tsne[:, 1], c=train_labels, cmap='viridis', alpha=0.6)
    plt.colorbar()
    plt.title('Training Embeddings Visualization (t-SNE)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    plt.subplot(1, 2, 2)
    plt.scatter(val_tsne[:, 0], val_tsne[:, 1], c=val_labels, cmap='viridis', alpha=0.6)
    plt.colorbar()
    plt.title('Validation Embeddings Visualization (t-SNE)')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    plt.tight_layout()
    plt.savefig('embeddings_visualization.png')

def classify():
    # Load the saved embeddings from training
    with open("train_embeddings.pkl", "rb") as f:
        train_embeddings, train_labels = pickle.load(f)

    # Load the model from modules.py
    model = SiameseNetwork().to(DEVICE)
    model.eval()  # Set model to evaluation mode

    # Load data loaders without sampler for now
    _, _, test_loader = get_data_loaders(batch_size=BATCH_SIZE)

    # If LOCAL, use a subset of the test dataset for faster testing
    if LOCAL:
        test_subset_indices = np.random.choice(len(test_loader.dataset), size=int(SUBSET_SIZE * len(test_loader.dataset)), replace=False)
        test_subset = Subset(test_loader.dataset, test_subset_indices)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the classifier with the reference set
    classifier = Classifier(margin=0.5)
    classifier.set_reference_set(torch.tensor(train_embeddings), torch.tensor(train_labels))

    # Lists to store predictions and true labels
    predictions = []
    true_labels = []

    # Iterate over the test set
    with torch.no_grad():
        for test_batch in tqdm(test_loader, desc="Testing"):
            test_img, test_label = test_batch
            test_img, test_label = test_img.to(DEVICE), test_label.to(DEVICE)

            # Get the embedding for the test image
            test_embedding = model(test_img).cpu()

            # Predict the class
            predicted_classes = [classifier.predict_class(emb) for emb in test_embedding]
            predictions.extend(predicted_classes)
            true_labels.extend(test_label.cpu().numpy().tolist())

    # Print summary of predictions
    correct_predictions = sum([pred == true for pred, true in zip(predictions, true_labels)])
    accuracy = correct_predictions / len(true_labels) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Plot ROC curve and Confusion Matrix
    from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
    import seaborn as sns

    # ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    sns.lineplot(x=fpr, y=tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')

    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    ConfusionMatrixDisplay(cm).plot()
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')


def main():
    # train()
    classify()

if __name__ == "__main__":
    main()