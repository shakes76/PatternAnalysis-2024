import os
import nibabel as nib
import gzip
import shutil

class NiftiDataset:
    def __init__(self, image_directory, label_directory, normImage=False, max_images=50):
        self.image_directory = image_directory
        self.label_directory = label_directory
        self.normImage = normImage
        
        self.image_files = [f for f in os.listdir(image_directory) if f.endswith('.gz')]
        self.label_files = [f for f in os.listdir(label_directory) if f.endswith('.gz')]
        assert len(self.image_files) == len(self.label_files), "The number of images and labels must match."
        self.image_files = self.image_files[:max_images]
        self.label_files = self.label_files[:max_images]

    def decompress_gz(self, file_path):
        decompressed_file = file_path.replace('.gz', '')
        with gzip.open(file_path, 'rb') as f_in:
            with open(decompressed_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return decompressed_file

    def load_data(self):
        images, labels = [], []
        for image_file, label_file in zip(self.image_files, self.label_files):
            decompressed_image_path = self.decompress_gz(os.path.join(self.image_directory, image_file))
            decompressed_label_path = self.decompress_gz(os.path.join(self.label_directory, label_file))

            img = nib.load(decompressed_image_path).get_fdata()
            label = nib.load(decompressed_label_path).get_fdata()

            # Normalize images if specified
            if self.normImage:
                img = (img - img.mean()) / img.std()

            label = label.astype(int)
            labels.append(label)

            images.append(img)

        return images, labels

# Example usage
dataset = NiftiDataset(image_path, label_path, normImage=True)
images, labels = dataset.load_data()
print(f"Loaded {len(images)} images and {len(labels)} labels.")
