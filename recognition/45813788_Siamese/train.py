import os
import torch
from torch.nn import BCEWithLogitsLoss
from modules import SiameseNN
from utils import visualise_embedding, plot_loss, plot_accuracy, plot_auc
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.distances import LpDistance
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np



def siamese_train(current_dir, train_loader, val_loader, epochs=50, lr=1e-4, plots=False):
    save_dir = os.path.join(current_dir, 'models')
    os.makedirs(save_dir, exist_ok=True)

    print("Training Siamese Network with Contrastive and BCEWithLogits Losses")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the Siamese network
    model = SiameseNN().to(device)

    # Loss functions
    criterion_bce = BCEWithLogitsLoss()
    criterion_contrastive = ContrastiveLoss(distance=LpDistance(normalize_embeddings=False))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.01)

    # Training parameters
    best_auroc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_aurocs = []
    val_aurocs = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0
        train_labels = []
        train_probs = []

        for images_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device).float()

            optimizer.zero_grad()

            # Forward pass
            embeddings, logits = model(images_batch)

            logits = logits.squeeze()

            # Compute losses
            loss_bce = criterion_bce(logits, labels_batch)
            loss_contrastive = criterion_contrastive(embeddings, labels_batch)

            # Total loss
            total_loss = loss_bce + loss_contrastive

            # Backward and optimize
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            # Predictions and probabilities
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            # Calculate correct predictions
            correct_train += (preds == labels_batch).sum().item()
            total_train += labels_batch.size(0)

            # Store labels and probabilities for AUROC
            train_labels.extend(labels_batch.cpu().numpy())
            train_probs.extend(probs.detach().cpu().numpy())

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Compute AUROC for training
        try:
            train_auroc = roc_auc_score(train_labels, train_probs)
            binary_train_preds = (np.array(train_probs) >= 0.5).astype(int)
            print("Training Classification Report:")
            print(classification_report(train_labels, binary_train_preds, zero_division=0))
        except ValueError:
            train_auroc = np.nan  # Handle cases where AUROC is undefined

        train_aurocs.append(train_auroc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_labels = []
        val_probs = []

        with torch.no_grad():
            for images_batch, labels_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validating"):
                images_batch, labels_batch = images_batch.to(device), labels_batch.to(device).float()

                embeddings, logits = model(images_batch)

                logits = logits.squeeze()

                # Compute losses
                loss_bce = criterion_bce(logits, labels_batch)
                loss_contrastive = criterion_contrastive(embeddings, labels_batch)

                # Total loss
                total_loss = loss_bce + loss_contrastive

                val_loss += total_loss.item()

                # Predictions and probabilities
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                # Calculate correct predictions
                correct_val += (preds == labels_batch).sum().item()
                total_val += labels_batch.size(0)

                # Store labels and probabilities for AUROC
                val_labels.extend(labels_batch.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        # Compute AUROC for validation
        try:
            val_auroc = roc_auc_score(val_labels, val_probs)
            binary_val_preds = (np.array(val_probs) >= 0.5).astype(int)
            print("Validation Classification Report:")
            print(classification_report(val_labels, binary_val_preds, zero_division=0))
        except ValueError:
            val_auroc = np.nan  # Handle cases where AUROC is undefined

        val_aurocs.append(val_auroc)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f} - Val Accuracy: {val_accuracy:.4f}")
        print(f"Train AUROC: {train_auroc:.4f} - Val AUROC: {val_auroc:.4f}")

        # Step the scheduler
        scheduler.step(avg_val_loss)

        # Monitor learning rate
        for idx, param_group in enumerate(optimizer.param_groups):
            print(f"Learning rate for param group {idx}: {param_group['lr']}")

        # Save the best model based on validation AUROC
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            save_path = os.path.join(save_dir, 'siamese_best.pth')
            torch.save(model.state_dict(), save_path)
            print("Validation AUROC improved. Saving model.")
        else:
            print("No improvement in validation AUROC.")

        # Visualize embeddings if needed
        if plots:
            all_embeddings = []
            with torch.no_grad():
                for images_batch, labels_batch in val_loader:
                    images_batch = images_batch.to(device)
                    embeddings, _ = model(images_batch)
                    all_embeddings.append(embeddings.cpu())

            all_embeddings_tensor = torch.cat(all_embeddings)
            visualise_embedding(all_embeddings_tensor, val_labels, epoch+1, current_dir)

    # Save the final model
    save_path = os.path.join(save_dir, 'siamese_final.pth')
    torch.save(model.state_dict(), save_path)
    print("Training complete. Models saved.")

    if plots:
        plot_loss(train_losses, val_losses)
        plot_accuracy(train_accuracies, val_accuracies)
        plot_auc(train_aurocs, val_aurocs)