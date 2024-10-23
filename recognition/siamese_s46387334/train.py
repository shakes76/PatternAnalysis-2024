"""
Contains the source code for training, validating, testing and saving the model. 

The model is imported from “modules.py” and the data loader is imported from “dataset.py”. 
Plots of the losses and metrics during training will be produced.
"""

###############################################################################
### Imports
from time import gmtime, strftime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score

from dataset import get_isic2020_data, get_isic2020_data_loaders
from modules import TripletLoss, SiameseNet, set_seed
from predict import predict_siamese_net, results_siamese_net


###############################################################################
### Config Settings
def get_config() -> dict:
    """
    Get the config used to train and evaluate SiameseNet on the ISIC 2020 data.

    Returns: the current config settings
    """
    config = {
        'data_subset': 1300,
        'metadata_path': '/kaggle/input/isic-2020-jpg-256x256-resized/train-metadata.csv',
        'image_dir': '/kaggle/input/isic-2020-jpg-256x256-resized/train-image/image/',
        'embedding_dims': 128,
        'learning_rate': 0.0001,
        'epochs': 2,
    }
    return config


###############################################################################
### Functions
def train_siamese_net(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: SiameseNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    triplet_loss: TripletLoss,
    classifier_loss: nn.Module,
    epochs: int,
    device: str
) -> None:
    """
    Runs the siamese net model through a training loop using the given optimiser, losses etc.
    The model will be trained using data from the train_loader and will be evaluated throughout training
    based on data from the val_loader.

    The model with the best AUR ROC on the validation set will be saved as siamese_net_model.pt 

    At the end of the training loop, graphs for the train and validation: loss, accuracy and AUR ROC over
    the epochs of training will be produced and saved to the folder.
    """
    best_val_aurroc = 0

    train_loss_per_epoch = []
    train_acc_per_epoch = []
    train_aucroc_per_epoch = []

    val_loss_per_epoch = []
    val_acc_per_epoch = []
    val_aucroc_per_epoch = []

    model.train()
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = []

        for _, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(train_loader):
            # Move the data to the device (GPU or CPU)
            anchor_img = anchor_img.to(device).float()
            positive_img = positive_img.to(device).float()
            negative_img = negative_img.to(device).float()
            anchor_label = anchor_label.to(device)

            optimizer.zero_grad()
            # Forward pass of net
            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)
            curr_trip_loss = triplet_loss(anchor_out, positive_out, negative_out)

            # Forward pass of classifier
            classifier_out = model.classify(anchor_img)
            curr_classifier_loss = classifier_loss(classifier_out, anchor_label)

            # Calculate loss
            loss = curr_trip_loss + curr_classifier_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss.append(loss.cpu().detach().numpy())

        # Calculate and print training loss
        avg_train_loss = np.mean(running_loss)

        # Validation phase (testing)
        model.eval()  # Set model to evaluation model
        with torch.no_grad():  # Disable gradient computation for validation/testing

            # Calculate Validation Loss
            val_running_loss = []
            for _, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(val_loader):
                anchor_img = anchor_img.to(device).float()
                positive_img = positive_img.to(device).float()
                negative_img = negative_img.to(device).float()
                anchor_label = anchor_label.to(device)
    
                # Loss from Forward pass of net
                anchor_out = model(anchor_img)
                positive_out = model(positive_img)
                negative_out = model(negative_img)
                curr_trip_loss = triplet_loss(anchor_out, positive_out, negative_out)
        
                # Loss from Forward pass of classifier
                classifier_out = model.classify(anchor_img)
                curr_classifier_loss = classifier_loss(classifier_out, anchor_label)
                
                # Calculate total loss
                val_loss = curr_trip_loss + curr_classifier_loss
                val_running_loss.append(val_loss.cpu().detach().numpy())
            avg_val_loss = np.mean(val_running_loss)
    
            # Calculate
            test_y_pred, test_y_probs, test_y_true, _ = predict_siamese_net(model, val_loader, device)
            val_accuracy = accuracy_score(test_y_true, test_y_pred)
            val_aucroc = roc_auc_score(test_y_true, test_y_probs)
        
            train_y_pred, train_y_probs, train_y_true, _ = predict_siamese_net(model, train_loader, device)
            train_accuracy = accuracy_score(train_y_true, train_y_pred)
            train_aucroc = roc_auc_score(train_y_true, train_y_probs)
        
            # Record current state for this epoch
            train_loss_per_epoch.append(avg_train_loss)
            val_loss_per_epoch.append(avg_val_loss)
            train_acc_per_epoch.append(train_accuracy)
            val_acc_per_epoch.append(val_accuracy)
            train_aucroc_per_epoch.append(train_aucroc)
            val_aucroc_per_epoch.append(val_aucroc)

        # Trigger scheduler based on the current Validation AUR ROC
        scheduler.step(val_aucroc)
        
        # Print out current results
        print(f"[{strftime('%H:%M:%S', gmtime())}] Epoch: {epoch+1 :>2}/{epochs} -- [Train Loss: {avg_train_loss:.4f} - Train Acc: {train_accuracy:.4f} - Train AUR ROC: {train_aucroc:.4f}] -- [Val Loss {avg_val_loss:.4f} - Val Acc: {val_accuracy:.4f} - Val AUC ROC: {val_aucroc:.4f}]")
    
        # Save the model if it preforms better than all other epochs on validation set
        if val_aucroc > best_val_aurroc:
            torch.save(model.state_dict(), "siamese_net_model.pt")
            best_val_aurroc = val_aucroc
            print(f"New model saved with Validation AUR ROC of: {best_val_aurroc:.4f}")

    # After Training is completed plot the progress
    plot_training_graphs(
        train_loss_per_epoch,
        val_loss_per_epoch,
        train_acc_per_epoch,
        val_acc_per_epoch,
        train_aucroc_per_epoch,
        val_aucroc_per_epoch,
        epochs
    )

def plot_training_graphs(
    train_loss_per_epoch,
    val_loss_per_epoch,
    train_acc_per_epoch,
    val_acc_per_epoch,
    train_aucroc_per_epoch,
    val_aucroc_per_epoch,
    epochs
) -> None:    
    """
    Plot graphs for the train and validation: loss, accuracy and AUR ROC over
    the epochs of training and save the plots to the folder.
    """
    # Plot Loss
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(epochs), train_loss_per_epoch, label='Train Loss', color='darkseagreen')
    plt.plot(range(epochs), val_loss_per_epoch, label='Validation Loss', color='lightcoral')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(range(epochs), train_acc_per_epoch, label='Train Accuracy', color='darkseagreen')
    plt.plot(range(epochs), val_acc_per_epoch, label='Validation Accuracy', color='lightcoral')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot AURROC
    plt.subplot(1, 3, 3)
    plt.plot(range(epochs), train_aucroc_per_epoch, label='Train AURROC', color='darkseagreen')
    plt.plot(range(epochs), val_aucroc_per_epoch, label='Validation AURROC', color='lightcoral')
    plt.title('AURROC over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('AURROC')
    plt.legend()
    
    # Show the plots
    plt.tight_layout()
    plt.show()


###############################################################################
### Main Function
def main() -> None:
    """
    Run Training on SiameseNet for classification of ISIC 2020 data.
    Training will be preformed and then evaluation results on the trained model will be produced.
    """   
    # Set Seed
    set_seed()

    # Determine device that we are training on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the current config
    config = get_config()
    
    # Extract the data from the given locations
    images, labels = get_isic2020_data(
        metadata_path=config['metadata_path'],
        image_dir=config['image_dir'],
        data_subset=config['data_subset']
    )

    # Get the data loaders
    train_loader, val_loader, test_loader = get_isic2020_data_loaders(images, labels)

    # Initalise Model
    model = SiameseNet(config['embedding_dims']).to(device)

    # Initialise loss functions
    triplet_loss = TripletLoss().to(device)
    classifier_loss = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # Initialise Optimiser and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # Train the model, best model (on validation set) will be saved to folder
    # Additionally loss, accuracy and aurroc plots will be produced and saved
    # To view the models progression over training
    train_siamese_net(
        train_loader,
        val_loader,
        model,
        optimizer,
        scheduler,
        triplet_loss,
        classifier_loss,
        config['epochs'],
        device
    )

    # Load in best model from the training
    model.load_state_dict(torch.load("siamese_net_model.pt", weights_only=False))
    
    # Predict results on the test set
    results_siamese_net(test_loader, model, device)


if __name__ == "__main__":
    main()