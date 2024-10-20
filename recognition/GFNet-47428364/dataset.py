import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

'''
Gets the list of all patient ID's from the data
'''
def get_patient_ids(data_path):
    # Files are encoded with the ID and the image number seperated by an underscore.
    all_files = [os.path.basename(file) for _, _, filenames in os.walk(data_path) for file in filenames]
    patient_ids = list(set([file.split('_')[0] for file in all_files]))
    return patient_ids

""" Returns the train and test dataloaders for the ADNI dataset """
def get_dataloaders(batch_size=32, path="recognition/GFNet-47428364/AD_NC"):
    # Create transformer
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Generate datasets
    train_dataset = ImageFolder(root=path+"/train", transform=transform)
    test_dataset = ImageFolder(root=path+"/test", transform=transform)

    patient_ids = get_patient_ids(path+"/train")
    train_indices = [i for i, (path_, _) in enumerate(train_dataset.samples) if path_.split('\\')[-1].split('_')[0] in patient_ids]
    train_subset = Subset(train_dataset, train_indices)
                              
    # Generate dataloaders
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader