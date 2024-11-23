import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from modules import SiameseNetwork
from dataset import Dataset
from predict import PredictData
import argparse

if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="A parser to get the path to the project folder and data folder")

    # Add arguments for project and data paths
    parser.add_argument("--project", type=str, help="The os path to the project directory")
    parser.add_argument("--data", type=str, help="The os path to the pre-processed data")

    # Parse the arguments
    args = parser.parse_args()

    # Set device to CUDA if a CUDA device is available, else CPU
    print(torch.cuda.is_available())  # Check if CUDA is available
    print(torch.cuda.device_count())  # Print the number of GPUs detected
    print(torch.cuda.get_device_name(0))  # Print the name of the GPU
    print(torch.version.cuda)  # Print CUDA version
    torch.cuda.empty_cache()  # Clear GPU cache
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parameter declaration
    learning_rate = 0.00001  # Learning rate for the optimizer
    num_epochs = 20  # Number of epochs to train the model
    backbone = "resnet18"  # The feature extraction model used in the Siamese network

    # General paths for project and data
    general_path = args.project  # Path to the project directory
    data_path = args.data  # Path to the data directory

    # Path to write outputs to (including checkpoints)
    out_path = os.path.join(general_path, "outputs")

    # Paths to training, testing, and validation data
    train_data_path = os.path.join(data_path, "train_data")
    test_data_path = os.path.join(data_path, "test_data")
    val_data_path = os.path.join(data_path, "validation_data")

    # Load training data
    train_data = Dataset(train_data_path)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=128, num_workers=0)
    print("Got training data")
    ############################################################
    # Load test data
    test_data = Dataset(test_data_path)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=0)
    print("Got test data")
    ############################################################
    # Load validation data
    val_data = Dataset(val_data_path)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=16, num_workers=0)
    print("Got validation data")
    ############################################################

    # Initialize the Siamese Network model
    model = SiameseNetwork()
    model.to(device)

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = torch.nn.BCELoss()  # Binary Cross-Entropy Loss

    # Print size of datasets as a sanity check
    print(f"Size of training data: {len(train_data)}")
    print(f"Size of validation data: {len(val_data)}")
    print(f"Size of test data: {len(test_data)}")

    # Lists to store training and validation accuracy/loss values for plotting
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Initialize best validation loss with a large value
    best_val = 10000000000

    # Training loop
    for epoch in range(num_epochs):
        print("[{} / {}]".format(epoch, num_epochs))
        # Set the model to training mode
        model.train()

        losses = []
        correct = 0
        total = 0

        # Training Loop Start
        for (img1, img2), target, (class1, class2) in train_data_loader:
            # Move images and target to the appropriate device
            img1, img2, target = map(lambda x: x.to(device), [img1, img2, target])

            # Get the similarity between the image pair
            similarity = model(img1, img2)
            # Calculate the loss
            loss = criterion(similarity, target)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Append loss and calculate accuracy
            losses.append(loss.item())
            correct += torch.count_nonzero(target == (similarity > 0.5)).item()
            total += len(target)

        # Calculate and store training loss and accuracy
        train_loss = sum(losses) / len(losses)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        print("\tTraining: Loss={:.2f}\t Accuracy={:.2f}\t".format(train_loss, train_acc))
        # Training Loop End

        # Evaluation Loop Start
        model.eval()  # Set the model to evaluation mode

        losses = []
        correct = 0
        total = 0

        # Validation loop
        for (img1, img2), target, (class1, class2) in val_data_loader:
            # Move images and target to the appropriate device
            img1, img2, target = map(lambda x: x.to(device), [img1, img2, target])

            # Get the similarity between the image pair
            similarity = model(img1, img2)
            # Calculate the loss
            loss = criterion(similarity, target)

            # Append loss and calculate accuracy
            losses.append(loss.item())
            correct += torch.count_nonzero(target == (similarity > 0.5)).item()
            total += len(target)

        # Calculate and store validation loss and accuracy
        val_loss = sum(losses) / max(1, len(losses))
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print("\tValidation: Loss={:.2f}\t Accuracy={:.2f}\t".format(val_loss, val_acc))
        # Evaluation Loop End

        # Save the model if the validation loss improves
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(out_path, "best.pth")
            )

    # Plot the training and validation accuracy and loss
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 4))

    # Plot for loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot for accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'training_validation_metrics.png'))  # Save the plot
    plt.show()

    # Run prediction on the test data using predict.py
    prediction = PredictData(test_data_loader, general_path)
    prediction.predict()
