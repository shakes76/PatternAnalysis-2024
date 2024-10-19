import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from dataset import ISICDataset, benign_aug, malig_aug

def test(siamese, classifier, test_df, images):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    siamese.to(device)
    siamese.eval()
    
    classifier.to(device)
    classifier.eval()

    test_dataset = ISICDataset(
    df=test_df,
    images_dir=images,
    #transform_benign=benign_aug,
    #transform_malignant=malig_aug,
    augment_ratio=0.0  
    )

    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32,
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    all_preds = []
    all_labels = []
    all_img_id = []

    with torch.no_grad():
        for images,labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            embeddings = siamese(images)
            output = classifier(embeddings).squeeze()

            preds = (output >= 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    print(f"Overall Test Accuracy: {accuracy:.4f}")

    # Print confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
            
