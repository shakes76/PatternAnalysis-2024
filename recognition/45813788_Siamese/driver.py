#calls and runs algo, everything from here
import os
from dataset import load_data, split_data
from train import siamese_train, classifier_train
from modules import SiameseNN, Classifier
import torch
from torch.utils.data import DataLoader
from predict import test
from pytorch_metric_learning.samplers import MPerClassSampler
from dataset import load_data, split_data, ISICDataset, benign_aug,malig_aug

# Run the training
def main():

    # Paths
    current_dir = os.getcwd()
    print("Working dir", current_dir)
    excel = os.path.join(current_dir, 'dataset', 'train-metadata.csv')
    images = os.path.join(current_dir, 'dataset', 'train-image', 'image')
    
    #make a directory to save models if not there
    save_dir = os.path.join(current_dir,'models')
    os.makedirs(save_dir, exist_ok=True)


    df = load_data(excel)
    train_df,val_df,test_df = split_data(df)

    #init loaders
        
    # Initialize training and validation datasets
    train_dataset = ISICDataset(
        df = train_df,
        images_dir=images,
        transform_benign=benign_aug,
        transform_malignant=malig_aug,
        augment_ratio=0.5
    )

    val_dataset = ISICDataset(
        df=val_df,
        images_dir=images,
        transform_benign=benign_aug,
        transform_malignant=malig_aug,
        augment_ratio=0.0  
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )


    siamese_train(current_dir, train_loader,val_loader, images, epochs=200, plots=False)

    siamese_net = SiameseNN()
    siamese_dict = os.path.join(current_dir, 'models', 'siamese_resnet18_best.pth')
    siamese_net.load_state_dict(torch.load(siamese_dict))
    classifier_train(current_dir, train_loader, val_loader, images, siamese_net, epochs=100, plots=True)

    #Testing part
    classifier_net = Classifier()
    classifier_dict = os.path.join(current_dir,'models','classifier_best.pth')
    classifier_net.load_state_dict(torch.load(classifier_dict)) 
    test(siamese_net,classifier_net,test_df,images)

    #we can improve this alot, lot of repeating code in training loops we can handle creating the DataLoaders here


if __name__ == "__main__":
    main()

