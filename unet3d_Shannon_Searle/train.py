# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from modules import UNet3D
from dataset import CustomDataset, Resize3D, Normalize3D

# Main function
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
        Resize3D((64, 64, 32)),  # Resize to (depth, height, width)
        Normalize3D()  # Normalize the images
    ])


    # Create dataset
    dataset= CustomDataset(image_filenames, img_dir, labels_dir, transform = transform)
    
    # Split into training and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # DataLoader for batching
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the 3D U-Net model
    model = UNet3D(in_channels=4, out_channels=6, init_features=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # Training loop
    n_epochs = 1
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}")
        model.train()
        running_loss = 0.0        
        for images, labels in train_loader:  # Unpack the tuple
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # Compute the loss (using labels as targets instead of images)
            labels_class_indices = torch.argmax(labels, dim=1)
            loss = criterion(outputs, labels_class_indices)  
            # Zero the gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(running_loss/len(train_loader))
    
    
if __name__ == "__main__":
    main()
