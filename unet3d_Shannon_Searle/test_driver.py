# test_driver.py
import os
import torch
from dataset import *
from torchvision import transforms
from torch.utils.data import DataLoader
from train import Trainer
from predict import evaluate_dice_on_test_set

os.environ['TQDM_DISABLE'] = 'True'

def main():
    # Construct file paths correctly
    img_dir = "Labelled_weekly_MR_images_of_the_male_pelvis-QEzDvqEq-/data/semantic_MRs_anon"
    labels_dir = "Labelled_weekly_MR_images_of_the_male_pelvis-QEzDvqEq-/data/semantic_labels_anon"
    # Check if the directories exist
    if not os.path.exists(img_dir):
        print(f"Image directory {img_dir} does not exist.")
    if not os.path.exists(labels_dir):
        print(f"Labels directory {labels_dir} does not exist.")
    
    # Proceed with data loading if paths are correct
    image_filenames = [f for f in os.listdir(img_dir) if f.endswith('.nii.gz')]
    
    # Define transformations
    transform = transforms.Compose([
        Resize3D((64, 64, 32)), 
        Normalize3D(),
        RandomFlip3D(axes=[0,1,2]), RandomRotate3D(angle_range=30)
    ])

    
    # Create dataset
    dataset= CustomDataset(image_filenames, img_dir, labels_dir, transform = transform)
    
    # Define proportions
    train_ratio = 0.7
    val_ratio = 0.15  # 15% for validation
    test_ratio = 0.15  # 15% for testing

    # Split into training, validation, and test sets
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    # Perform the splits
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    # DataLoader for batching
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Train the model
    print("Starting model training...")
    trainer = Trainer(train_loader, val_loader)
    trained_model = trainer.train(n_epochs=20)
    
    # Test the model
    print("\nStarting model testing...")
    avg_dice_score, min_coeff = evaluate_dice_on_test_set(trained_model, test_loader)
    print(f"Average Dice Similarity Coefficient on Test Set: {avg_dice_score:.4f}")
    print(f"All Labels had a Minimum Similarity Coefficient of 0.7: {min_coeff}")

    
if __name__ == "__main__":
    main()