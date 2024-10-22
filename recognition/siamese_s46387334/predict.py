import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

def results_siamese_net(
    test_loader,
    model: SiameseNet,
    device
):    
    model.eval()
    with torch.no_grad():
        # Plot of the spread of the test data points in the embedded space
        test_labels = []
        test_results = []

        for img, _, _, label in test_loader:
            test_results.append(model(img.float().to(device)).cpu().numpy())
            test_labels.append(label)
            
        test_results = np.concatenate(test_results)
        test_labels = np.concatenate(test_labels)
        
        plt.figure(figsize=(15, 10), facecolor="azure")
        for label in np.unique(test_labels):
            tmp = test_results[test_labels==label]
            plt.scatter(tmp[:, 0], tmp[:, 1], label=label, alpha=0.4)
        
        plt.legend()
        plt.show()
    
        # Results from the testing set
        test_y_pred, test_y_probs, test_y_true, test_embeddings = predict_siamese_net(model, test_loader, device)
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

        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(test_embeddings)
    
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=test_y_true)
        plt.colorbar(scatter)
        plt.title('t-SNE visualization of embeddings')
        plt.show()

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
    all_embeddings = []

    for batch_idx, (imgs, _, _, y_true) in enumerate(data_loader):
        imgs = imgs.to(device).float()
        outputs = model.classify(imgs) 

        # Determine model embeddings
        embeddings = model(imgs)

        # Determine positive class probability
        y_prob = torch.softmax(outputs, dim=1)[:, 1]

        # Determine the predicted class
        _, y_pred = outputs.max(1)

        all_y_pred.extend(y_pred.cpu().numpy())
        all_y_prob.extend(y_prob.cpu().numpy())
        all_y_true.extend(y_true.cpu().numpy())
        all_embeddings.extend(embeddings.cpu().numpy())

    return np.array(all_y_pred), np.array(all_y_prob), np.array(all_y_true), np.array(all_embeddings)


###############################################################################
### Main Function
def main():
    """
    If we call this modual directly
    """
    # Set Seed
    set_seed()
    
    # Determine device that we are testing on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the current config
    config = get_config()
    
    # Load in best model - we will use this model for our final predictions on test set
    model = SiameseNet(config['embedding_dims']).to(device)
    model.load_state_dict(torch.load("siamese_net_model.pt", weights_only=False))

    # Extract the data from the given locations
    images, labels = get_isic2020_data(
        metadata_path=config['metadata_path'],
        image_dir=config['image_dir'],
        data_subset=config['data_subset']
    )

    # Get the data loaders
    _, _, test_loader = get_isic2020_data_loaders(images, labels)

    # Determine results of best model on test set
    results_siamese_net(test_loader, model,  device)

# if __name__ == "__main__":
#     main()
