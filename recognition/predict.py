import torch
from train import Train
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np




def main():
    # call train
    training = Train()
    train_loss_list = training.start_training()
    # plot loss graph
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_list[0], label='Training Loss', color='blue', marker='o')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(np.arange(len(train_loss_list[0])))  # Ensure x-ticks match the number of epochs
    plt.grid()
    plt.legend()
    plt.show()

    # evaulte
    average_iou, accuracy = training.evaluate()
    print("Average IOU: ", average_iou)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()