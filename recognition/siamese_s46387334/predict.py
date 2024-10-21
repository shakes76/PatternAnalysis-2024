"""
Shows example usage of the trained model.

Any results and / or visualisations will be printed / saved.
"""


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def results_siamese_net(
    test_loader,
    model: SiameseNet
):
    model.eval()
    with torch.no_grad():
        for img, _, _, label in train_loader:
            train_results.append(model(img.float().to(device)).cpu().numpy())
            labels.append(label)
            
    train_results = np.concatenate(train_results)
    labels = np.concatenate(labels)
    train_results.shape
    
    plt.figure(figsize=(15, 10), facecolor="azure")
    for label in np.unique(labels):
        tmp = train_results[labels==label]
        plt.scatter(tmp[:, 0], tmp[:, 1], label=label, alpha=0.4)
    
    plt.legend()
    plt.show()

    test_y_pred, test_y_probs, test_y_true, test_embeddings = predict(model, test_loader)
    test_accuracy = accuracy_score(test_y_true, test_y_pred)
    test_auc_roc = roc_auc_score(test_y_true, test_y_probs)

    print(f"Testing Accuracy: {test_accuracy}")
    print(f"Testing AUR ROC: {test_auc_roc}")

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(test_y_true, test_y_pred)

    # Normalize the confusion matrix by rows (i.e., by the actual class counts)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Plot the normalized confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2%', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.title('Confusion Matrix (Percentages)')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.show()



###############################################################################
### Main Function
def main():
    """
    If we call this modual directly
    """   
    # Determine device that we are training on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load in best model - we will use this model for our final predictions on test set
    model = SiameseNet(CONFIG['embedding_dims']).to(device)
    model.load_state_dict(torch.load("siamese_net_model.pt"))

    # Extract the data from the given locations
    images, labels = get_isic2020_data(
        metadata_path=CONFIG['metadata_path'],
        image_dir=CONFIG['image_dir'],
        data_subset=CONFIG['data_subset']
    )

    # Get the data loaders
    _, _, test_loader = get_isic2020_data_loaders(images, labels)

    # Determine results of best model on test set
    results_siamese_net(test_loader, model)

# if __name__ == "__main__":
#     main()
