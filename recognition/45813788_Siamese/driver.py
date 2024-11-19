#calls and runs algo, everything from here
import os
from dataset import load_data, split_data
from train import siamese_train
from modules import SiameseNN
import torch
from torch.utils.data import DataLoader
from predict import test
from pytorch_metric_learning.samplers import MPerClassSampler
from dataset import load_data, split_data, ISICDataset, train_aug
from hyper import *
import argparse



def main():

    '''
    function that runs the whole thing, training and testing
    '''
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train or Test a Siamese Neural Network")
    parser.add_argument('-m', choices=['train', 'test','both'], default='both', required=True, help="Choose whether to train or test the model, you must train before testing")
    parser.add_argument('-p', type=bool, default=False, help="Enable or disable plotting during training (True/False)")
    args = parser.parse_args()

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
        transform = train_aug,
        augment_ratio=0.5
    )

    val_dataset = ISICDataset(
        df=val_df,
        images_dir=images,
        transform=train_aug,
        augment_ratio=0.0  
    )

    test_dataset = ISICDataset(
    df=test_df,
    images_dir=images,
    augment_ratio=0.0  
    )


    #balance samples
    labels = train_dataset.labels
    sampler = MPerClassSampler(labels, m=(BATCH_SIZE/2), batch_size=BATCH_SIZE)


    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        sampler=sampler, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    if args.m == 'train' or args.m == 'both':
        #Train the siamese
        siamese_train(current_dir, train_loader, val_loader, epochs=EPOHCS, lr=LEARNING_RATE, plots=args.p)

    if args.m == 'test' or args.m == 'both':
        #Test it
        siamese_net = SiameseNN()
        siamese_dict = os.path.join(current_dir, 'models', 'siamese_best.pth')
        siamese_net.load_state_dict(torch.load(siamese_dict))

        test(siamese_net, test_loader, current_dir)


if __name__ == "__main__":
    main()