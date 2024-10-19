#calls and runs algo, everything from here
import os
from dataset import load_data, split_data
from train import siamese_train



# Run the training
def main():
    # Paths
    current_dir = os.getcwd()
    print("Working dir", current_dir)
    excel = os.path.join(current_dir, 'dataset', 'train-metadata.csv')
    images = os.path.join(current_dir, 'dataset', 'train-image', 'image')

    df = load_data(excel)
    train_df,val_df,test_df = split_data(df)
    siamese_train(current_dir, train_df,val_df, images, plots=True)

if __name__ == "__main__":
    main()

