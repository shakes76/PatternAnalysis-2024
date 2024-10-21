"""
Contains the source code for training, validating, testing and saving the model. 

The model is imported from “modules.py” and the data loader is imported from “dataset.py”. 
Plots of the losses and metrics during training will be produced.
"""

###############################################################################
### Imports
from time import gmtime, strftime
import torch.optim as optim
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#from dataset import get_isic2020_data, get_isic2020_data_loaders
#from modules import TripletLoss, SiameseNet




###############################################################################
### Functions
def predict_siamese_net(
    model: SiameseNet,
    data_loader: DataLoader,
    device
) -> (list, list, list):
    """
    """
    all_y_pred = []
    all_y_prob = []
    all_y_true = []

    for batch_idx, (imgs, _, _, labels) in enumerate(data_loader):
        imgs = imgs.to(device).float()
        outputs = model.classify(imgs)   

        # Determine positive class probability
        y_prob = torch.softmax(outputs, dim=1)[:, 1]

        # Determine the predicted class
        _, y_pred = outputs.max(1)

        all_y_pred.extend(y_pred.cpu().numpy())
        all_y_prob.extend(y_prob.cpu().numpy())
        all_y_true.extend(labels.cpu().numpy())

    return np.array(all_y_pred), np.array(all_y_prob), np.array(all_y_true)

def train_siamese_net(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: SiameseNet,
    optimizer,
    triplet_loss: TripletLoss,
    classifier_loss,
    epochs: int,
    device
) -> None:
    """
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
        
        for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(train_loader):
            # Move the data to the device (GPU or CPU)
            anchor_img = anchor_img.to(device).float()
            positive_img = positive_img.to(device).float()
            negative_img = negative_img.to(device).float()
            anchor_label = anchor_label.to(device)
            
            optimizer.zero_grad()
            # Foward pass of net
            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)
            curr_trip_loss = triplet_loss(anchor_out, positive_out, negative_out)
    
            # Foward pass of classifer
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
            for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(val_loader):
                anchor_img = anchor_img.to(device).float()
                positive_img = positive_img.to(device).float()
                negative_img = negative_img.to(device).float()
                anchor_label = anchor_label.to(device)
    
                # Loss from Foward pass of net
                anchor_out = model(anchor_img)
                positive_out = model(positive_img)
                negative_out = model(negative_img)
                curr_trip_loss = triplet_loss(anchor_out, positive_out, negative_out)
        
                # Loss from Foward pass of classifer
                classifier_out = model.classify(anchor_img)
                curr_classifier_loss = classifier_loss(classifier_out, anchor_label)
                
                # Calculate total loss
                val_loss = curr_trip_loss + curr_classifier_loss
                val_running_loss.append(val_loss.cpu().detach().numpy())
            avg_val_loss = np.mean(val_running_loss)
    
            # Calculate
            test_y_pred, test_y_probs, test_y_true = predict_siamese_net(model, val_loader, device)
            val_accuracy = accuracy_score(test_y_true, test_y_pred)
            val_aucroc = roc_auc_score(test_y_true, test_y_probs)
        
            train_y_pred, train_y_probs, train_y_true = predict_siamese_net(model, train_loader, device)
            train_accuracy = accuracy_score(train_y_true, train_y_pred)
            train_aucroc = roc_auc_score(train_y_true, train_y_probs)
        
            # Record current state for this epoch
            train_loss_per_epoch.append(avg_train_loss)
            val_loss_per_epoch.append(avg_val_loss)
            train_acc_per_epoch.append(train_accuracy)
            val_acc_per_epoch.append(val_accuracy)
            train_aucroc_per_epoch.append(train_aucroc)
            val_aucroc_per_epoch.append(val_aucroc)
    
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
    )

def plot_training_graphs(
    train_loss_per_epoch,
    val_loss_per_epoch,
    train_acc_per_epoch,
    val_acc_per_epoch,
    train_aucroc_per_epoch,
    val_aucroc_per_epoch,
):    
    """
    """
    # Plot Loss
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(range(EPOCHS), train_loss_per_epoch, label='Train Loss', color='darkseagreen')
    plt.plot(range(EPOCHS), val_loss_per_epoch, label='Validation Loss', color='lightcoral')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(range(EPOCHS), train_acc_per_epoch, label='Train Accuracy', color='darkseagreen')
    plt.plot(range(EPOCHS), val_acc_per_epoch, label='Validation Accuracy', color='lightcoral')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot AURROC
    plt.subplot(1, 3, 3)
    plt.plot(range(EPOCHS), train_aucroc_per_epoch, label='Train AURROC', color='darkseagreen')
    plt.plot(range(EPOCHS), val_aucroc_per_epoch, label='Validation AURROC', color='lightcoral')
    plt.title('AURROC over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('AURROC')
    plt.legend()
    
    # Show the plots
    plt.tight_layout()
    plt.show()

def set_seed(seed: int=42):
    """
    """
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


###############################################################################
### Main Function
def main():
    """
    """   
    # Determine device that we are training on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract the data from the given locations
    images, labels = get_isic2020_data(
        metadata_path=CONFIG['metadata_path'],
        image_dir=CONFIG['image_dir'],
        data_subset=CONFIG['data_subset']
    )

    # Get the data loaders
    train_loader, val_loader, test_loader = get_isic2020_data_loaders(images, labels)

    # Initalise Model
    model = SiameseNet(CONFIG['embedding_dims']).to(device)

    # Initialise loss fucnctions and optimiser
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    triplet_loss = TripletLoss().to(device)
    classifier_loss = nn.CrossEntropyLoss().to(device)

    # Train the model, best model (on validation set) will be saved to folder
    train_siamese_net(
        train_loader,
        val_loader,
        model,
        optimizer,
        triplet_loss,
        classifier_loss,
        CONFIG['epochs'],
        device
    )

    # Predict results on testing
    results_siamese_net

    # Predict Results from test set

if __name__ == "__main__":
    main()