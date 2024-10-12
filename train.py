import os
import glob
import random
import torch
import numpy as np
import torchio as tio
import torch.nn as nn
import torch.optim as optim
from modules import UNet3D
from dataset import NiftiDataset
from torch.utils.data import DataLoader


def train_loop():
    # Define device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        print("Warning: No access to CUDA, model being trained on CPU.")
    
    # Paths to your images and labels
    images_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
    labels_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

    # Gather all image filenames
    image_filenames = glob.glob(os.path.join(images_path, '*.nii*'))
    image_filenames.sort()  # Ensure consistent ordering

    # Gather all label filenames
    label_filenames = glob.glob(os.path.join(labels_path, '*.nii*'))
    label_filenames.sort()  # Ensure consistent ordering

    print(f"Found {len(image_filenames)} images and {len(label_filenames)} labels.")

    # Extract basenames (filenames without directory paths)
    image_basenames = [os.path.basename(f) for f in image_filenames]
    label_basenames = [os.path.basename(f) for f in label_filenames]

    # Align filenames based on a common identifier
    def get_subject_id(filename):
        basename = os.path.basename(filename)
        basename_no_ext = basename.split('.')[0]
        subject_id = '_'.join(basename_no_ext.split('_')[:-1])  # Remove the last part after '_'
        return subject_id

    # Create dictionaries mapping subject IDs to filenames
    image_dict = {get_subject_id(f): f for f in image_filenames}
    label_dict = {get_subject_id(f): f for f in label_filenames}

    # Find common subject IDs
    common_subject_ids = set(image_dict.keys()).intersection(set(label_dict.keys()))
    matched_image_filenames = [image_dict[sid] for sid in sorted(common_subject_ids)]
    matched_label_filenames = [label_dict[sid] for sid in sorted(common_subject_ids)]

    print(f"Matched {len(matched_image_filenames)} image-label pairs.")

    # Update the filenames lists
    image_filenames = matched_image_filenames
    label_filenames = matched_label_filenames

    # Combine image and label filenames into pairs
    data_pairs = list(zip(image_filenames, label_filenames))

    # Set a seed for reproducibility
    random.seed(42)
    random.shuffle(data_pairs)

    # Define split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # Calculate the number of samples for each set
    total_samples = len(data_pairs)
    train_count = int(total_samples * train_ratio)
    val_count = int(total_samples * val_ratio)
    test_count = total_samples - train_count - val_count  # Adjust for any rounding errors

    # Split the data pairs
    train_pairs = data_pairs[:train_count]
    val_pairs = data_pairs[train_count:train_count + val_count]
    test_pairs = data_pairs[train_count + val_count:]

    # Unzip the pairs back into separate lists
    train_image_filenames, train_label_filenames = zip(*train_pairs)
    val_image_filenames, val_label_filenames = zip(*val_pairs)
    test_image_filenames, test_label_filenames = zip(*test_pairs)

    print(f"Training samples: {len(train_image_filenames)}")
    print(f"Validation samples: {len(val_image_filenames)}")
    print(f"Test samples: {len(test_image_filenames)}")

    # Define transforms
    training_transform = tio.Compose([
        tio.RandomAffine(),
        tio.RandomElasticDeformation(),
        tio.RandomFlip(axes=(0, 1, 2)),
        tio.RandomNoise(),
        # TODO Add more transforms ?
    ])

    # Training dataset
    train_dataset = NiftiDataset(
        image_filenames=train_image_filenames,
        label_filenames=train_label_filenames,
        normImage=True,
        categorical=False,
        dtype=np.float32,
        getAffines=False,
        orient=False,
        transform=None
    )

    # Validation dataset
    val_dataset = NiftiDataset(
        image_filenames=val_image_filenames,
        label_filenames=val_label_filenames,
        normImage=True,
        categorical=False,
        dtype=np.float32,
        getAffines=False,
        orient=False,
        transform=None
    )

    # Test dataset
    test_dataset = NiftiDataset(
        image_filenames=test_image_filenames,
        label_filenames=test_label_filenames,
        normImage=True,
        categorical=False,
        dtype=np.float32,
        getAffines=False,
        orient=False,
        transform=None
    )

    batch_size = 1  # Adjust based on your GPU memory
    num_workers = 1  # Adjust based on your CPU cores

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    model = UNet3D().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # TODO learning scheduler?

    num_epochs = 15
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        for batch_images, batch_labels in train_loader:

            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            # Remove the channels dimension from labels
            batch_labels = batch_labels.squeeze(1)  # Shape becomes (batch_size, D, H, W)

            # Ensure labels are of type torch.long
            # batch_labels = batch_labels.long().permute(0, 4, 1, 2, 3)

            optimizer.zero_grad()
            outputs = model(batch_images)  # Should output shape: (batch_size, num_classes, D, H, W)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)
                batch_labels = batch_labels.squeeze(1)  # Shape becomes (batch_size, D, H, W)
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item() * batch_images.size(0)

        epoch_val_loss = val_loss / len(val_dataset)
        print(f"Validation Loss: {epoch_val_loss:.4f}")

    # Save the trained model
    # torch.save(model.state_dict(), 'unet_model.pth')
    # print('Training complete and model saved.')
    print('Training Complete.')

    # Testing phase
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            batch_labels = batch_labels.squeeze(1)
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            test_loss += loss.item() * batch_images.size(0)

    avg_test_loss = test_loss / len(test_dataset)
    print(f"Test Loss: {avg_test_loss:.4f}")

if __name__ == "__main__":
    train_loop()

# import os
# import glob
# import random
# import torch
# import numpy as np
# import torchio as tio
# import torch.nn as nn
# import torch.optim as optim
# from modules import UNet3D
# from dataset import NiftiDataset
# from torch.utils.data import DataLoader
# from torch.optim import lr_scheduler
# from tqdm import tqdm
# import copy

# def train_loop(
#     images_path="/home/groups/comp3710/HipMRI_Study_open/semantic_MRs",
#     labels_path="/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only",
#     batch_size=2,
#     num_workers=1,
#     num_epochs=50,
#     learning_rate=1e-4,
#     patience=10,  # For early stopping
#     save_path='best_unet_model.pth'
# ):
#     # Define device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     if not torch.cuda.is_available():
#         print("Warning: No access to CUDA, model being trained on CPU.")

#     # Gather all image and label filenames
#     image_filenames = sorted(glob.glob(os.path.join(images_path, '*.nii*')))
#     label_filenames = sorted(glob.glob(os.path.join(labels_path, '*.nii*')))
#     print(f"Found {len(image_filenames)} images and {len(label_filenames)} labels.")

#     # Function to extract subject ID for alignment
#     def get_subject_id(filename):
#         basename = os.path.basename(filename)
#         basename_no_ext = basename.split('.')[0]
#         subject_id = '_'.join(basename_no_ext.split('_')[:-1])  # Adjust based on your naming convention
#         return subject_id

#     # Create dictionaries mapping subject IDs to filenames
#     image_dict = {get_subject_id(f): f for f in image_filenames}
#     label_dict = {get_subject_id(f): f for f in label_filenames}

#     # Find common subject IDs
#     common_subject_ids = sorted(set(image_dict.keys()).intersection(set(label_dict.keys())))
#     matched_image_filenames = [image_dict[sid] for sid in common_subject_ids]
#     matched_label_filenames = [label_dict[sid] for sid in common_subject_ids]
#     print(f"Matched {len(matched_image_filenames)} image-label pairs.")

#     # Combine image and label filenames into pairs
#     data_pairs = list(zip(matched_image_filenames, matched_label_filenames))

#     # Set a seed for reproducibility
#     random.seed(42)
#     random.shuffle(data_pairs)

#     # Define split ratios
#     train_ratio = 0.7
#     val_ratio = 0.15
#     test_ratio = 0.15

#     # Calculate the number of samples for each set
#     total_samples = len(data_pairs)
#     train_count = int(total_samples * train_ratio)
#     val_count = int(total_samples * val_ratio)
#     test_count = total_samples - train_count - val_count  # Adjust for any rounding errors

#     # Split the data pairs
#     train_pairs = data_pairs[:train_count]
#     val_pairs = data_pairs[train_count:train_count + val_count]
#     test_pairs = data_pairs[train_count + val_count:]

#     # Unzip the pairs back into separate lists
#     train_image_filenames, train_label_filenames = zip(*train_pairs)
#     val_image_filenames, val_label_filenames = zip(*val_pairs)
#     test_image_filenames, test_label_filenames = zip(*test_pairs)

#     print(f"Training samples: {len(train_image_filenames)}")
#     print(f"Validation samples: {len(val_image_filenames)}")
#     print(f"Test samples: {len(test_image_filenames)}")

#     # Initialize datasets with transforms
#     train_dataset = NiftiDataset(
#         image_filenames=train_image_filenames,
#         label_filenames=train_label_filenames,
#         normImage=True,  # Normalization handled by transforms
#         categorical=False,  # Assuming multi-class segmentation
#         dtype=np.float32,
#         getAffines=False,
#         orient=False,
#         transform=None
#     )

#     val_dataset = NiftiDataset(
#         image_filenames=val_image_filenames,
#         label_filenames=val_label_filenames,
#         normImage=True,
#         categorical=False,
#         dtype=np.float32,
#         getAffines=False,
#         orient=False,
#         transform=None
#     )

#     test_dataset = NiftiDataset(
#         image_filenames=test_image_filenames,
#         label_filenames=test_label_filenames,
#         normImage=True,
#         categorical=False,
#         dtype=np.float32,
#         getAffines=False,
#         orient=False,
#         transform=None
#     )

#     # Define DataLoaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=num_workers,
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#     )

#     # Initialize the model, criterion, optimizer, and scheduler
#     model = UNet3D().to(device)
#     criterion = nn.CrossEntropyLoss()  # Suitable for multi-class segmentation
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

#     # Initialize variables for early stopping and best model
#     best_val_loss = float('inf')
#     best_model_wts = copy.deepcopy(model.state_dict())
#     epochs_no_improve = 0

#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch+1}/{num_epochs}")
#         print("-" * 10)

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#                 dataloader = train_loader
#             else:
#                 model.eval()   # Set model to evaluate mode
#                 dataloader = val_loader

#             running_loss = 0.0
#             running_dice = 0.0  # For Dice Coefficient

#             # Iterate over data
#             loop = tqdm(dataloader, desc=f"{phase.capitalize()} Iteration", leave=False)
#             for batch_images, batch_labels in loop:
#                 batch_images = batch_images.to(device)
#                 batch_labels = batch_labels.to(device)

#                 # Forward pass
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(batch_images)  # Shape: (batch_size, num_classes, D, H, W)
#                     loss = criterion(outputs, batch_labels)

#                     # Calculate Dice Coefficient
#                     preds = torch.argmax(outputs, dim=1)
#                     dice = dice_coefficient(preds, batch_labels, num_classes=outputs.shape[1])

#                     # Backward and optimize only in training phase
#                     if phase == 'train':
#                         optimizer.zero_grad()
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * batch_images.size(0)
#                 running_dice += dice * batch_images.size(0)

#                 # Update progress bar
#                 loop.set_postfix(loss=loss.item(), dice=dice.item())

#             epoch_loss = running_loss / len(dataloader.dataset)
#             epoch_dice = running_dice / len(dataloader.dataset)

#             print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Dice Coef: {epoch_dice:.4f}")

#             # Deep copy the model if it's the best so far
#             if phase == 'val':
#                 scheduler.step(epoch_loss)  # Step the scheduler based on validation loss
#                 if epoch_loss < best_val_loss:
#                     best_val_loss = epoch_loss
#                     best_model_wts = copy.deepcopy(model.state_dict())
#                     torch.save(best_model_wts, save_path)
#                     print(f"Best model saved with val loss: {best_val_loss:.4f}")
#                     epochs_no_improve = 0
#                 else:
#                     epochs_no_improve += 1
#                     print(f"No improvement for {epochs_no_improve} epochs.")

#         # Early stopping
#         if epochs_no_improve >= patience:
#             print("Early stopping triggered.")
#             break

#     # Load best model weights
#     model.load_state_dict(best_model_wts)
#     print("Training complete. Best validation loss: {:.4f}".format(best_val_loss))

#     # Optionally, evaluate on the test set
#     evaluate_model(model, test_loader, device)

# def dice_coefficient(preds, labels, num_classes):
#     """
#     Calculate Dice Coefficient for each class and return the average.
#     """
#     smooth = 1e-5
#     dice = 0.0
#     for cls in range(num_classes):
#         pred_cls = (preds == cls).float()
#         label_cls = (labels == cls).float()
#         intersection = (pred_cls * label_cls).sum()
#         union = pred_cls.sum() + label_cls.sum()
#         dice += (2. * intersection + smooth) / (union + smooth)
#     return dice / num_classes

# def evaluate_model(model, dataloader, device):
#     """
#     Evaluate the model on the test set and print metrics.
#     """
#     model.eval()
#     test_loss = 0.0
#     test_dice = 0.0
#     criterion = nn.CrossEntropyLoss()
#     with torch.no_grad():
#         for batch_images, batch_labels in tqdm(dataloader, desc="Testing"):
#             batch_images = batch_images.to(device)
#             batch_labels = batch_labels.to(device)

#             outputs = model(batch_images)
#             loss = criterion(outputs, batch_labels)
#             test_loss += loss.item() * batch_images.size(0)

#             preds = torch.argmax(outputs, dim=1)
#             dice = dice_coefficient(preds, batch_labels, num_classes=outputs.shape[1])
#             test_dice += dice.item() * batch_images.size(0)

#     avg_loss = test_loss / len(dataloader.dataset)
#     avg_dice = test_dice / len(dataloader.dataset)
#     print(f"\nTest Loss: {avg_loss:.4f} | Test Dice Coef: {avg_dice:.4f}")

# if __name__ == "__main__":
#     train_loop()
