"""
Contains the source code for training, validating, testing and saving the model. 

The model is imported from “modules.py” and the data loader is imported from “dataset.py”. 
Plots of the losses and metrics during training will be produced.
"""

###############################################################################
### Imports
from dataset import get_isic2020_data, get_isic2020_data_loaders


###############################################################################
### Functions
def predict_siamese_net(model, data_loader, device):
    """
    """
    all_y_pred = []
    all_y_prob = []
    all_y_true = []
    
    for batch_idx, (imgs, _, _, labels) in enumerate(data_loader):
        imgs = imgs.to(device).float()
        outputs = model.classify(imgs)   
        
        # Determine positive class probability
        y_prob = torch.softmax(outputs, dim=1)[:, 1] 

        # Determine the predicted class
        _, y_pred = outputs.max(1)
        
        all_y_pred.extend(y_pred.cpu().numpy())
        all_y_prob.extend(y_prob.cpu().numpy())
        all_y_true.extend(labels.cpu().numpy())
    
    return np.array(all_y_pred), np.array(all_y_prob), np.array(all_y_true)

###############################################################################
### Main Function
def main():
    """
    """
    config = {
        'data_subset': 100,
        'metadata_path': '/kaggle/input/isic-2020-jpg-256x256-resized/train-metadata.csv',
        'image_dir': '/kaggle/input/isic-2020-jpg-256x256-resized/train-image/image/',
    }

    # Extract the data from the given locations
    images, labels = get_isic2020_data(
        metadata_path=config['metadata_path'],
        image_dir=config['image_dir'],
        data_subset=config['data_subset']
    )

    # Get the data loaders
    train_loader, val_loader, test_loader = get_isic2020_data_loaders(images, labels)


if __name__ == "__main__":
    main()
