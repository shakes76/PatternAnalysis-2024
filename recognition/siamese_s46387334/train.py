"""
Contains the source code for training, validating, testing and saving the model. 

The model is imported from “modules.py” and the data loader is imported from “dataset.py”. 
Plots of the losses and metrics during training will be produced.
"""

###############################################################################
### Imports
from dataset import get_isic2020_data, get_isic2020_data_loaders


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
