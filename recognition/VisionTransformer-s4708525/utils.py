import matplotlib.pyplot as plt

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