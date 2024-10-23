import matplotlib.pyplot as plt
import numpy as np

def show_plot_loss(train_losses, val_losses):
    """
    Function to handle the visualization of the training and validation loss

    Args:
        train_losses: list of training loss
        val_losses: list of validation loss
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/train_val_loss_001.png", bbox_inches='tight')

def show_plot_accuracy(train_acc, val_acc):
    """
    Function to handle the visualization of the training and validation accuracy

    Args:
        train_acc: list of training accuracy
        val_acc: list of validation accuracy
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/train_val_accuracy_001.png", bbox_inches='tight')

def evaluate_prediction(predicted_class, num):
    """
    Function to evaluate prediction results

    Args:
        num: the result of prediction, 0 for NC, 1 for AD
    """
    if predicted_class == num:
        results = "Correct predictions!"
    else:
        results = "Incorrect predictions!"

    return results

def imshow(img):
    """
    Function to unnormalize and convert tensor to numpy array for display

    Args:
        img: image on the tensor format
    """
    img = img.numpy().transpose((1, 2, 0)) 
    mean = np.array([0.1155, 0.1155, 0.1155]) 
    std = np.array([0.2224, 0.2224, 0.2224]) 
    img = std * img + mean 
    img = np.clip(img, 0, 1)
    return img


def show_sample_data(train_loader, classes, file_name):
    """
    Visualize a batch of images from the DataLoader

    Args:
        train_loader: the data loader to be visualized
        classes: the class/label of the dataset
        file_name: name of the file to be saved
    """
    # Get a batch of training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # Create a 2x8 grid for 16 images
    fig, axes = plt.subplots(2, 8, figsize=(20, 6))

    for i, ax in enumerate(axes.flat):
        img = images[i]
        label = labels[i].item()

        ax.imshow(imshow(img))
        ax.set_title(f"Label: {classes[label]}")
        ax.axis('off') 

    plt.tight_layout() 
    plt.savefig(f"results/{file_name}.png", bbox_inches='tight')