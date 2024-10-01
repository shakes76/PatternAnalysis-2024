import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def get_data_loaders(train_dir, test_dir, batch_size=32, val_ratio=0.2, random_seed=42):
    # Define transforms for training and validation/test
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Load the dataset without any transforms initially
    full_dataset = datasets.ImageFolder(root=train_dir, transform=None)
    
    # Mapping from patient ID to image indices
    patient_to_indices = {}
    for idx, (path, _) in enumerate(full_dataset.samples):
        filename = os.path.basename(path)
        patient_id = filename.split('_')[0]
        if patient_id not in patient_to_indices:
            patient_to_indices[patient_id] = []
        patient_to_indices[patient_id].append(idx)
    
    # List of all unique patient IDs
    all_patient_ids = list(patient_to_indices.keys())
    
    # Split patient IDs into train and validation sets
    train_patient_ids, val_patient_ids = train_test_split(
        all_patient_ids,
        test_size=val_ratio,
        random_state=random_seed,
        shuffle=True
    )
    
    # Gather all image indices for training and validation
    train_indices = []
    for pid in train_patient_ids:
        train_indices.extend(patient_to_indices[pid])
    
    val_indices = []
    for pid in val_patient_ids:
        val_indices.extend(patient_to_indices[pid])

    # Create subsets for training and validation
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    # Assign transforms
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_test_transform
    
    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load test dataset with validation/test transforms
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader




