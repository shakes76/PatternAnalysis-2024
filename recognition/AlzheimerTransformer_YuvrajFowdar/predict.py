# Function to create the plots
def save_plots(train_losses, val_losses, train_accuracies, val_accuracies, save_dir="plots"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # Ensure plots directory exists
    
    # Plotting loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss_plot.png")  # Save the plot
    plt.close()  # Close the figure to free memory
    
    # Plotting accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/accuracy_plot.png")  # Save the plot
    plt.close()