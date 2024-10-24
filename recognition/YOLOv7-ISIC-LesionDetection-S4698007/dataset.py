import os  
import nibabel as nib  
import gzip  
import shutil  
import torch  
from torch.utils.data import Dataset, DataLoader  

class NiftiDataset(Dataset):
    """
    class for loading NIfTI images and their corresponding labels.

    Attributes:
        image_directory (str): Path to NIfTI images.
        label_directory (str): Path to NIfTI labels.
        normImage (bool): Whether to normalize the images.
        max_images (int): Maximum number of images to load from the directories.
        image_files (list): List of gzipped image filenames.
        label_files (list): List of gzipped label filenames.
    """

    def __init__(self, image_directory, label_directory, normImage=False, max_images=50):
        """
        Initializes the NiftiDataset instance.

        Args:
            image_directory (str): Path to zipped images.
            label_directory (str): Path to the zipped labels.
            normImage (bool): Flag indicating whether to normalize the images.
            max_images (int): Maximum number of images to load.
        """
        self.image_directory = image_directory  # Store image directory path
        self.label_directory = label_directory  # Stor the label directory path
        self.normImage = normImage 
        
        # List all gzipped image files in the image directory
        self.image_files = [f for f in os.listdir(image_directory) if f.endswith('.gz')]
        # List all gzipped label files in the label directory
        self.label_files = [f for f in os.listdir(label_directory) if f.endswith('.gz')]

        # Ensure the number of images matches the number of labels
        assert len(self.image_files) == len(self.label_files), "No. of images and labels must match."

        # Limit the number of files loaded to max_images
        self.image_files = self.image_files[:max_images]
        self.label_files = self.label_files[:max_images]

    def decompress_gz(self, file_path):
        """
        Decompress a gzipped file.

        Args:
            file_path (str): The path to the gzipped file.

        Returns:
            str: The path to the decompressed file.
        """
        # Create the path for the decompressed file by removing the .gz extension
        decompressed_file = file_path.replace('.gz', '')
        # Open the gzipped file for reading and the decompressed file for writing
        with gzip.open(file_path, 'rb') as f_in:
            with open(decompressed_file, 'wb') as f_out:
                # Copy the contents from the gzipped file to the decompressed file
                shutil.copyfileobj(f_in, f_out)
        return decompressed_file  # Return the path of the decompressed file

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_files)  # Return the length of the image files list

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

        # Convert the image and label data to PyTorch tensors
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add a channel dimension for grayscale images
        label_tensor = torch.tensor(label, dtype=torch.long)  # Convert labels to long type for classification

        return img_tensor, label_tensor  # Return the image and label tensors

def create_dataloader(image_directory, label_directory, batch_size=4, max_images=50, normImage=True):
    """
    Create a DataLoader for the NiftiDataset.

    Args:
        image_directory (str): Path to the directory containing gzipped images.
        label_directory (str): Path to the directory containing gzipped labels.
        batch_size (int): Number of samples per batch to load.
        max_images (int): Maximum number of images to load.
        normImage (bool): Flag indicating whether to normalize the images.

    Returns:
        DataLoader: A DataLoader object for the dataset.
    """
    # Create an instance of the NiftiDataset with the provided parameters
    dataset = NiftiDataset(image_directory, label_directory, normImage, max_images)
    # Return a DataLoader for the dataset, which allows for easy batch processing and shuffling
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Shuffle the data for each epoch
