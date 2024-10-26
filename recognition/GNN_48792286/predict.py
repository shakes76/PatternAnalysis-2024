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
# Visualization with t-SNE
def visualize(model, data):
    model.eval()
    with torch.no_grad():
        z = model(data).detach().numpy()
        # Perform t-SNE dimensionality reduction with modified parameters
        z = TSNE(n_components=2, perplexity=10, max_iter=500, random_state=None).fit_transform(z)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=data.y.numpy(), cmap='viridis', alpha=0.7)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Class Label')
    plt.title('t-SNE Visualization of GNN Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

