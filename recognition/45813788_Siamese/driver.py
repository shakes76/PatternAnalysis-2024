#calls and runs algo, everything from here
import os
from dataset import load_data, split_data
from train import siamese_train, classifier_train
from modules import SiameseNN, Classifier
import torch
from predict import test


# Run the training
def main():
    # Paths
    current_dir = os.getcwd()
    print("Working dir", current_dir)
    excel = os.path.join(current_dir, 'dataset', 'train-metadata.csv')
    images = os.path.join(current_dir, 'dataset', 'train-image', 'image')

    df = load_data(excel)
    train_df,val_df,test_df = split_data(df)
    siamese_train(current_dir, train_df,val_df, images, epochs=200, plots=True)

    siamese_net = SiameseNN()
    siamese_dict = os.path.join(current_dir, 'models', 'siamese_resnet18_best.pth')
    siamese_net.load_state_dict(torch.load(siamese_dict))
    classifier_train(current_dir, train_df, val_df, images, siamese_net, epochs=100, plots=True)

    #Testing part
    classifier_net = Classifier()
    classifier_dict = os.path.join(current_dir,'models','classifier_best.pth')
    classifier_net.load_state_dict(torch.load(classifier_dict))
    test(siamese_net,classifier_net,test_df,images)

    #we can improve this alot, lot of repeating code in training loops we can handle creating the DataLoaders here


if __name__ == "__main__":
    main()

