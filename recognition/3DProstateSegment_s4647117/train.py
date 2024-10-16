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


# def train_loop():
#     # Define device
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     if not torch.cuda.is_available():
#         print("Warning: No access to CUDA, model being trained on CPU.")
    
#     # Paths to your images and labels
#     images_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
#     labels_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

#     # Gather all image filenames
#     image_filenames = glob.glob(os.path.join(images_path, '*.nii*'))
#     image_filenames.sort()  # Ensure consistent ordering

#     # Gather all label filenames
#     label_filenames = glob.glob(os.path.join(labels_path, '*.nii*'))
#     label_filenames.sort()  # Ensure consistent ordering

#     print(f"Found {len(image_filenames)} images and {len(label_filenames)} labels.")

#     # Extract basenames (filenames without directory paths)
#     image_basenames = [os.path.basename(f) for f in image_filenames]
#     label_basenames = [os.path.basename(f) for f in label_filenames]

#     # Align filenames based on a common identifier
#     def get_subject_id(filename):
#         basename = os.path.basename(filename)
#         basename_no_ext = basename.split('.')[0]
#         subject_id = '_'.join(basename_no_ext.split('_')[:-1])  # Remove the last part after '_'
#         return subject_id

#     # Create dictionaries mapping subject IDs to filenames
#     image_dict = {get_subject_id(f): f for f in image_filenames}
#     label_dict = {get_subject_id(f): f for f in label_filenames}

#     # Find common subject IDs
#     common_subject_ids = set(image_dict.keys()).intersection(set(label_dict.keys()))
#     matched_image_filenames = [image_dict[sid] for sid in sorted(common_subject_ids)]
#     matched_label_filenames = [label_dict[sid] for sid in sorted(common_subject_ids)]

#     print(f"Matched {len(matched_image_filenames)} image-label pairs.")

#     # Update the filenames lists
#     image_filenames = matched_image_filenames
#     label_filenames = matched_label_filenames

#     # Combine image and label filenames into pairs
#     data_pairs = list(zip(image_filenames, label_filenames))

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

#     # Define transforms
#     training_transform = tio.Compose([
#         tio.RandomAffine(),
#         tio.RandomElasticDeformation(),
#         tio.RandomFlip(axes=(0, 1, 2)),
#     ])

#     # Training dataset
#     train_dataset = NiftiDataset(
#         image_filenames=train_image_filenames,
#         label_filenames=train_label_filenames,
#         normImage=True,
#         categorical=False,
#         dtype=np.float32,
#         getAffines=False,
#         orient=False,
#         transform=None
#     )

#     # Validation dataset
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

#     # Test dataset
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

#     batch_size = 1  # Adjust based on your GPU memory
#     num_workers = 1  # Adjust based on your CPU cores

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,  # Shuffle training data
#         num_workers=num_workers
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers
#     )

#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers
#     )

#     model = UNet3D().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4) # TODO learning scheduler?

#     num_epochs = 15
#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch+1}/{num_epochs}")

#         # Training phase
#         model.train()
#         running_loss = 0.0
#         for batch_images, batch_labels in train_loader:

#             batch_images = batch_images.to(device)
#             batch_labels = batch_labels.to(device)

#             # Remove the channels dimension from labels
#             batch_labels = batch_labels.squeeze(1)  # Shape becomes (batch_size, D, H, W)

#             # Ensure labels are of type torch.long
#             # batch_labels = batch_labels.long().permute(0, 4, 1, 2, 3)

#             optimizer.zero_grad()
#             outputs = model(batch_images)  # Should output shape: (batch_size, num_classes, D, H, W)
#             loss = criterion(outputs, batch_labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * batch_images.size(0)

#         epoch_loss = running_loss / len(train_dataset)
#         print(f"Training Loss: {epoch_loss:.4f}")

#         # Validation phase
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch_images, batch_labels in val_loader:
#                 batch_images = batch_images.to(device)
#                 batch_labels = batch_labels.to(device)
#                 batch_labels = batch_labels.squeeze(1)  # Shape becomes (batch_size, D, H, W)
#                 outputs = model(batch_images)
#                 loss = criterion(outputs, batch_labels)
#                 val_loss += loss.item() * batch_images.size(0)

#         epoch_val_loss = val_loss / len(val_dataset)
#         print(f"Validation Loss: {epoch_val_loss:.4f}")

#     # Save the trained model
#     # torch.save(model.state_dict(), 'unet_model.pth')
#     # print('Training complete and model saved.')
#     print('Training Complete.')

#     # Testing phase
#     model.eval()
#     test_loss = 0.0
#     with torch.no_grad():
#         for batch_images, batch_labels in test_loader:
#             batch_images = batch_images.to(device)
#             batch_labels = batch_labels.to(device)
#             batch_labels = batch_labels.squeeze(1)
#             outputs = model(batch_images)
#             loss = criterion(outputs, batch_labels)
#             test_loss += loss.item() * batch_images.size(0)

#     avg_test_loss = test_loss / len(test_dataset)
#     print(f"Test Loss: {avg_test_loss:.4f}")

# if __name__ == "__main__":
#     train_loop()


import os
import glob
import random
import torch
import numpy as np
import torchio as tio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import nibabel as nib
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
        tio.RandomFlip(axes=(0, 1, 2)),
    ])

    # Training dataset
    train_dataset = NiftiDataset(
        image_filenames=train_image_filenames,
        label_filenames=train_label_filenames,
        dtype=np.float32,
        transform=None
    )

    # Validation dataset
    val_dataset = NiftiDataset(
        image_filenames=val_image_filenames,
        label_filenames=val_label_filenames,
        dtype=np.float32,
        transform=None
    )

    # Test dataset
    test_dataset = NiftiDataset(
        image_filenames=test_image_filenames,
        label_filenames=test_label_filenames,
        dtype=np.float32,
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


    def dice_loss(pred, target, epsilon=1e-6):
        pred_probs = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3)
        target_one_hot = target_one_hot.type(pred.dtype)
        pred_flat = pred_probs.view(pred_probs.shape[0], pred_probs.shape[1], -1)
        target_flat = target_one_hot.view(target_one_hot.shape[0], target_one_hot.shape[1], -1)
        intersection = (pred_flat * target_flat).sum(-1)
        union = pred_flat.sum(-1) + target_flat.sum(-1)
        dice_score = (2 * intersection + epsilon) / (union + epsilon)
        dice_loss = 1 - dice_score.mean()
        return dice_loss

    model = UNet3D().to(device)
    # criterion = nn.CrossEntropyLoss()
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
            loss = dice_loss(outputs, batch_labels)
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
                loss = dice_loss(outputs, batch_labels)
                val_loss += loss.item() * batch_images.size(0)

        epoch_val_loss = val_loss / len(val_dataset)
        print(f"Validation Loss: {epoch_val_loss:.4f}")

    # Save the trained model
    # torch.save(model.state_dict(), 'unet_model.pth')
    # print('Training complete and model saved.')
    print('Training Complete.')

    # Testing phase
    is_first = True
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            batch_labels = batch_labels.squeeze(1)
            outputs = model(batch_images)
            if is_first:
                pred_class = torch.argmax(outputs, dim=1)
                pred_class = pred_class.squeeze(0)
                pred_class = pred_class.detach().cpu().numpy().astype(np.int16)
                pred_image = nib.Nifti1Image(pred_class, np.eye(4))
                nib.save(pred_image, "predicted_seg.nii.gz")

                actual = batch_labels.squeeze(0)
                actual = actual.detach().cpu().numpy().astype(np.int16)
                actual = nib.Nifti1Image(actual, np.eye(4))
                nib.save(actual, "actual_seg.nii.gz")

                is_first = False

            loss = dice_loss(outputs, batch_labels)
            test_loss += loss.item() * batch_images.size(0)

    avg_test_loss = test_loss / len(test_dataset)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Average Dice Score: {1 - avg_test_loss:.4f}")

if __name__ == "__main__":
    train_loop()


"""
#TODO:
- Add header blocks (@author tag)
- Add references in the ReadMe (Dice Loss?, 3DUnet paper?)
- Add Comments throughout and deleted uneeded comments
- Check if you need to use softmax in the model
- Generate plots with TensorBoard for the ReadMe
- ReadMe:
    - File structure
    - Example input/labels
    - Explaining the usage
    - Declare hyperparameters
"""
