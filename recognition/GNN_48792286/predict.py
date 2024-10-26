def plot_loss_accuracy(losses, accuracies):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(losses, color='tab:blue', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:red')
    ax2.plot(accuracies, color='tab:red', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Loss and Accuracy over Epochs')
    plt.show()

