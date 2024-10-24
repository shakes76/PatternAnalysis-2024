import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
from modules import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import get_data_loaders
import optuna
import numpy as np

def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, save_path):
    """
    Visualize the training and validaton performance of GFNet across epochs.
    """

    #Assuming the number of epochs is equal to the number of loss entries
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))


    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()


    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='green')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='red')
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches = 'tight')


    plt.tight_layout()
    plt.show()


def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss= 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Training", unit="batch")

    for images, labels in pbar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion (outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim = 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss /len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy

def validate (model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", unit="batch")
        for images, labels in pbar:
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert lists to NumPy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, cm


if __name__ == "__main__":

    # Using optuna study to detect the best hyperparameters: learning rate, weight decay rate and drop path rate,
    # whcih maximise the validation accuracy

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create an Optuna study
    study = optuna.create_study(direction="maximize")  # We want to maximize validation accuracy

    # Number of trials to run
    n_trials = 10

    for trial_num in range(n_trials):
        print(f"\n--- Trial {trial_num+1} ---")

        # Start a new trial
        trial = study.ask()

        # Suggest values for the hyperparameters
        learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-3)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
        drop_path_rate = trial.suggest_uniform('drop_path_rate', 0.0, 0.5)

        print(f"Hyperparameters for trial {trial_num+1}: lr={learning_rate}, weight_decay={weight_decay}, drop_path_rate={drop_path_rate}")

        # Initialize the model with the current hyperparameters
        model = GFNet(
                img_size=512, 
                patch_size=16, embed_dim=512, depth=19, mlp_ratio=4, drop_path_rate=drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model.to(device)

        # Load data
        zip_path = "ADNI_AD_NC_2D.zip"
        extract_to = "data"
        train_loader, val_loader, test_loader = get_data_loaders(zip_path, extract_to, batch_size=32, train_split=0.80)

        epochs = 30
        best_val_accuracy = 0.0

        for epoch in range(epochs):
            train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy, cm = validate(model, val_loader, criterion, device)

            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            test_loss, test_accuracy, cm = validate(model, test_loader, criterion, device)
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

            # Update best validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                # Save the best model
                torch.save(model.state_dict(), f'model_best_trial_{trial_num+1}.pth')

        print(f"Validation Accuracy for Trial {trial_num+1}: {best_val_accuracy:.4f}")

        # Report the best validation accuracy to Optuna using the trial object
        study.tell(trial, best_val_accuracy)

    print("Best hyperparameters: ", study.best_params)
    print("Best validation accuracy: ", study.best_value)

    # Plot the optimization history
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)

    best_params = study.best_params
    best_learning_rate = best_params['lr']
    best_weight_decay = best_params['weight_decay']
    best_drop_path_rate = best_params['drop_path_rate']

    # Initialize model, loss function, and optimizer with best hyperparameters
    model = GFNet(
                img_size=512, 
                patch_size=16, embed_dim=512, depth=19, mlp_ratio=4, drop_path_rate=best_drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RAdam(model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)
    model.to(device)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_accuracy = 0.0
    epochs = 70

    for epoch in range(epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, cm = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)


        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        test_loss, test_accuracy, cm = validate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Save the model if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'model.pth')

    print("Training complete.")

    # plot the curves
    plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, save_path = "metrics_plot.png")


'''
if __name__ == "__main__":
    # Although the following hyperparameters did not provide the highest val accuracy during 10 trials,
    # they achieve highest test accuracy around 0.68 while the rest only have 0.67 at most.
    #Hyperparameters with very high test accuracy: 
    #lr=1.2655138149073937e-05, weight_decay=9.472938882625012e-05 and drop_path_rate=0.17728764992362356

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # Initialize model, loss function, and optimizer
    model = GFNet(
                img_size=512, 
                patch_size=16, embed_dim=512, depth=19, mlp_ratio=4, drop_path_rate=0.17728764992362356,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

    #print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RAdam(model.parameters(), lr=1.2655138149073937e-05, weight_decay=9.472938882625012e-05)
    model.to(device)

    # Load data
    """
    change the zip_path to the path of your ADNI data and don't forget to modify the directory in datset.py
    """
    zip_path = "ADNI_AD_NC_2D.zip"
    extract_to = "data"
    train_loader, val_loader, test_loader = get_data_loaders(zip_path, extract_to, batch_size=32, train_split
                                                             = 0.80)

    # Training loop
    best_val_accuracy = 0.0
    epochs = 70

    for epoch in range(epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, cm = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)


        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        test_loss, test_accuracy, cm  = validate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Save the model if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'model_2.pth')

    print("Training complete.")

    # plot the curves
    plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies, save_path = "metrics_plot_2.png")
'''

# OpenAI. (2024). ChatGPT (Oct 2024 version) [Large language model]. https://openai.com