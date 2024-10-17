import torch
from torch.utils.data import DataLoader
from dataset import ISISCDataset, malig_aug, benign_aug
from modules import SiameseNN
import pandas as pd
import os
from pytorch_metric_learning.losses import ContrastiveLoss
from sklearn.model_selection import train_test_split

def load_data(excel):

    df = pd.read_csv(excel)

    df= df.drop(columns=['Unnamed: 0'])
    df= df.drop(columns=['patient_id'])

    benign = df[df['target'] == 0].reset_index(drop=True)
    malignant = df[df['target'] == 1].reset_index(drop=True)

    return malignant, benign

def siamese_train():

    #paths
    dir = os.getcwd()
    current_dir = os.getcwd()
    excel = os.path.join(current_dir,'recognition','45813788_Siamese','dataset', 'train-metadata.csv')
    images = os.path.join(current_dir,'recognition','45813788_Siamese','dataset', 'train-image','image')


    malignant_df, benign_df = load_data(excel=excel)

    #now i need sklearn for test train split beacuse i have to stratify since there arent many malignant samples even with agumentation
    benign_train, benign_val = train_test_split(benign_df,test_size=0.1, stratify=benign_df['target'], random_state=42)
    malignant_train, malignant_val = train_test_split(malignant_df,test_size=0.1, stratify=malignant_df['target'], random_state=42)

    #Intitialise training and testing for siamese
    train_dataset = ISISCDataset(benign_df=benign_train,malignant_df=malignant_train,
                                 images_dir=images,
                                 transform_benign=benign_aug, transform_malignant=malig_aug,
                                 augment_ratio=0.5) #idk if i need this but will do for now
    
    #Intitialise training and testing for siamese
    val_dataset = ISISCDataset(benign_df=benign_val,malignant_df=malignant_val,
                                 images_dir=images,
                                 transform_benign=benign_aug, transform_malignant=malig_aug,
                                 augment_ratio=0.5) #idk if i need this but will do for now
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=128,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    
    val_loader = DataLoader(val_dataset, 
                              batch_size=128,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    
    #Init goodies 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNN(embedding_dim=256).to(device)

    #contrastive Loss
    #deafults from webpage look good: https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#contrastiveloss 
    contrastive_loss = ContrastiveLoss()