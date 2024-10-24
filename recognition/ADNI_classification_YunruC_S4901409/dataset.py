import os
import zipfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

def extract_zip(zip_path, extract_to):
    '''Extracts the zip file into the data folder.'''
    if not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.") 

class ADNIDataset(Dataset):
    '''Custom dataset class inherited from pytorch dataset class.
        It is used to load and preprocess the dataset'''
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # Load data for both classes (AD and NC)
        self._load_data('AD', 1)
        self._load_data('NC', 0)

    def _load_data(self, folder_name, label):
        folder_path = os.path.join(self.data_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist.")
            return
        
        # Add all images from the given folder to the dataset
        for file in os.listdir(folder_path):
            if file.endswith('.jpeg'):
                img_path = os.path.join(folder_path, file)
                self.data.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def __getinfo__(self):
        # counting number of AD and NC cases in the ADNI dataset
        count_0 = sum(1 for label in self.labels if label == 0)
        count_1 = sum(1 for label in self.labels if label == 1)
        return len(self.labels), count_0, count_1

def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()

    mean /= num_pixels
    std /= num_pixels

    return mean, std
    
if __name__ == "__main__":
    #CALCULATE THE  mean and std values of DANI dataset
    zip_path = "ADNI_AD_NC_2D.zip"
    extract_to = "data"

    if zip_path.endswith('.zip'):
        extract_zip(zip_path, extract_to)
        data_dir = extract_to
    else: 
        data_dir = zip_path


    data_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    batch_size = 32

    train_dataset = ADNIDataset(os.path.join(data_dir, 'AD_NC/train'), transform = data_transforms)
    test_dataset = ADNIDataset(os.path.join(data_dir, 'AD_NC/test'), transform= data_transforms)

    loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    mean_tr, std_tr = get_mean_std(loader_train)
    mean_te, std_te = get_mean_std(loader_test)

    print ("Train_mean: ", mean_tr , ", Train_std: ", std_tr)
    print ("test_mean: ", mean_te , ", test_std: ", std_te)
    
def get_data_loaders(zip_path, extract_to, batch_size=32, train_split = 0.80):
    """ Loading the training, testing and validating dataset.
    The training and validating dataset are seperated from 
    the original train dataset"""

    # for situation if the images are contained in a zip folder 
    if zip_path.endswith('.zip'):
        extract_zip(zip_path, extract_to)
        data_dir = extract_to
    else: 
        data_dir = zip_path

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=7.2927e-08, std=1.3933e-07)
        ])
    
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.05),
        transforms.RandomVerticalFlip(0.10),
        transforms.RandomResizedCrop(size=(512, 512), scale=(1.05, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=7.2043e-08, std=1.3888e-07)
    ])
    
    """
    You can change the 'AD_NC/train' and 'AD_NC/test' 
    to the actual folder that contain the AD and NC images if necessary
    """
    train_dataset = ADNIDataset(os.path.join(data_dir, 'AD_NC/train'), transform= train_transform)
    test_dataset = ADNIDataset(os.path.join(data_dir, 'AD_NC/test'), transform= transform)

    train_size = int(train_split*len(train_dataset))
    val_size = len(train_dataset)-train_size

    train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False, num_workers=4)

    return train_loader, val_loader, test_loader
    


if __name__ == "__main__":
    '''
    Testing whether the function works or not
    You can adjust the following path to the path where the data is stored
    '''
    
    zip_path = "ADNI_AD_NC_2D.zip"
    extract_to = "data"

    train_loader, val_loader, test_loader = get_data_loaders(zip_path, extract_to)


    # test whetehr the loader works 
    for loader, name in zip([train_loader, val_loader, test_loader], ["Train", "Val", "Test"]):
        print(f"\nTesting {name} loader:")
        for images, labels in loader:
            print(f"Batch size: {len(images)}")
            print(f"Image shape: {images.shape}") 
            print(f"Labels: {labels}")
            break 
            
# OpenAI. (2024). ChatGPT (Oct 2024 version) [Large language model]. https://openai.com
