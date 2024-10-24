import os  
import nibabel as nib  
import gzip  
import shutil  
import torch  
from torch.utils.data import Dataset, DataLoader  
import torchvision.transforms as transforms  # Import transforms for resizing

class NiftiDataset(Dataset):
    """
    Class for loading NIfTI images and their corresponding labels.
    """
    def __init__(self, image_directory, label_directory, normImage=False, max_images=50):
        self.image_directory = image_directory  
        self.label_directory = label_directory  
        self.normImage = normImage 
        
        # List all gzipped image files
        self.image_files = [f for f in os.listdir(image_directory) if f.endswith('.gz')]
        # List all gzipped label files
        self.label_files = [f for f in os.listdir(label_directory) if f.endswith('.gz')]

        # Ensure the number of images matches the number of labels
        assert len(self.image_files) == len(self.label_files), "No. of images and labels must match."

        # Limit the number of files loaded to max_images
        self.image_files = self.image_files[:max_images]
        self.label_files = self.label_files[:max_images]

        # Define the transform for resizing
        self.resize_transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert the tensor to PIL Image
            transforms.Resize((256, 128)),  # Resize to 256x128
            transforms.ToTensor()  # Convert back to tensor
        ])

    def decompress_gz(self, file_path):
        """Decompress a gzipped file."""
        decompressed_file = file_path.replace('.gz', '')
        with gzip.open(file_path, 'rb') as f_in:
            with open(decompressed_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return decompressed_file  

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_files)  

    def __getitem__(self, idx):
        """
        Load and return an image and its corresponding label as tensors.

        Args:
            idx (int): Index of the item to fetch.

        Returns:
            tuple: A tuple containing the image tensor and the label tensor.
        """
        # Get the current image and label filenames using the provided index
        image_file = self.image_files[idx]
        label_file = self.label_files[idx]

        # Construct full paths for the compressed image and label files
        compressed_image_path = os.path.join(self.image_directory, image_file)
        compressed_label_path = os.path.join(self.label_directory, label_file)

        # Decompress the image and label files
        decompressed_image_path = self.decompress_gz(compressed_image_path)
        decompressed_label_path = self.decompress_gz(compressed_label_path)

        # Load the decompressed NIfTI images into numpy arrays
        img = nib.load(decompressed_image_path).get_fdata()  # Load image data
        label = nib.load(decompressed_label_path).get_fdata()  # Load label data

        # Normalize the image data if the normalization flag is set
        if self.normImage:
            img = (img - img.mean()) / img.std()  # Standard score normalization

        # Convert the label data to integers for classification
        label = label.astype(int)
        # Apply a threshold to the label data to create binary labels
        label[label >= 6] = 0  # Set all labels greater than or equal to 6 to 0

        # Ensure the image has the correct shape (C, H, W)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add a channel dimension for grayscale images
        label_tensor = torch.tensor(label, dtype=torch.long)  # Convert label to tensor

        # Resize the image and label to 256x128
        img_tensor = transforms.Resize((256, 128))(img_tensor)  # Resize the image
        label_tensor = transforms.Resize((256, 128))(label_tensor.unsqueeze(0)).squeeze(0)  # Resize labels and remove extra dimension

        return img_tensor, label_tensor  # Return the image and label tensors
 

def create_dataloader(image_directory, label_directory, batch_size=4, max_images=1000000, normImage=True):
    """Create a DataLoader for the NiftiDataset."""
    dataset = NiftiDataset(image_directory, label_directory, normImage, max_images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)  
