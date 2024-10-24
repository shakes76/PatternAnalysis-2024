# dataset.py
import os
import nibabel as nib
import gzip
import shutil

class NiftiDataset:
    def __init__(self, image_directory, label_directory, max_images=50):
        self.image_directory = image_directory
        self.label_directory = label_directory
        
        # Gather all gzipped image and label files
        self.image_files = [f for f in os.listdir(image_directory) if f.endswith('.gz')]
        self.label_files = [f for f in os.listdir(label_directory) if f.endswith('.gz')]

        # Ensure that the number of images and labels match
        assert len(self.image_files) == len(self.label_files), "The number of images and labels must match."

        # Limit the number of files to load
        self.image_files = self.image_files[:max_images]
        self.label_files = self.label_files[:max_images]

    def decompress_gz(self, file_path):
        """Decompress a .gz file."""
        decompressed_file = file_path.replace('.gz', '')
        with gzip.open(file_path, 'rb') as f_in:
            with open(decompressed_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return decompressed_file

    def load_data(self):
        """Load NIfTI images and labels."""
        images, labels = [], []
        for image_file, label_file in zip(self.image_files, self.label_files):
            # Decompress the images and labels
            decompressed_image_path = self.decompress_gz(os.path.join(self.image_directory, image_file))
            decompressed_label_path = self.decompress_gz(os.path.join(self.label_directory, label_file))

            # Load NIfTI images
            img = nib.load(decompressed_image_path).get_fdata()
            label = nib.load(decompressed_label_path).get_fdata()
            
            images.append(img)
            labels.append(label)

        return images, labels

# Example usage
image_path = r"C:\Users\sophi\Downloads\HipMRI_study_keras_slices_data\keras_slices_train"
label_path = r"C:\Users\sophi\Downloads\HipMRI_study_keras_slices_data\keras_slices_seg_train"
dataset = NiftiDataset(image_path, label_path)
images, labels = dataset.load_data()
print(f"Loaded {len(images)} images and {len(labels)} labels.")
