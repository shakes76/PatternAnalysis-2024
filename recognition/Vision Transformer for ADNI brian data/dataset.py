from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchdata.datapipes.iter import BucketBatcher, FileLister
from PIL import Image
import os.path as osP

## Using dataset path from rangpur, assuming this will be used for assessing as well
## Structured like below
## - DATASET_PATH/
    ## - test/
        ##  - AD/ (contains 4450 images)
        ##  - NC/ (contains 4540 images)
    ## - train/
        ##  - AD/ (contains 10400 images)
        ##  - NC/ (contains 11120 images)
DATASET_PATH = '/home/groups/comp3710/ADNI/AD_NC'
## Start with generic batch size of 32, can change depending on model training procedure & results
BATCH_SIZE = 32
## Images naming convention is PatientID__MRISliceID.jpeg total of 20 images per patient
IMAGES_PER_PATIENT = 20
## Current image size is 256 x 240, rezise to 224 x 224 to better suit convolutions
IMAGE_SIZE = 224

## Define dataset transform
### send image to tensor
### basic resize & crop for convolutions
### basic normalisation for RGB inesity values per channel 
#### use 0.5 to place intensity values between [-1, 1]
DATASET_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
    transforms.ToTensor()
    ])

"""
Method to apply transform to single image
"""
def apply_tf(image_data, tf=DATASET_TRANSFORM):
    image, class_name = image_data
    return tf(image), class_name

def open_image(data):
    filename, class_name = data
    return Image.open(filename).convert("RGB"), class_name

"""
Method to determine the class label for the given file based on its filename.

Params:
    filename (str): file name for given input image
Returns:
    filename (str): file name for given input image
    class_name (int): the class for the image, 0 = AD, 1 = NC
Throws an exception if the class label cannot be determined from provided filename
"""
def label_file(filename):
    split = filename.split("AD_NC")
    if split[-1].find("AD") != -1:
        class_name = 0
    elif split[-1].find("NC") != -1:
        class_name = 1
    else:
        return Exception()
    return filename, class_name

"""
Sorts the provided selection (bucket) of images based on filenames, such that 
images of the same patient are grouped in the same batch

Assumes all image filenames within a bucket are from the same directory, to ensure
correct lexicographical sorting order. Grouping is on patient ID.

Params:
    bucket (torch object): a collection (bucket) of images, including filenames
Returns:
    bucket (torch object): the bucket after having been sorted by filename, in 
    lexicographical order   
"""
def sort_patients(bucket):
    return sorted(bucket)

def load_data(path=DATASET_PATH, transform=DATASET_TRANSFORM, batch_size=BATCH_SIZE,
               imgs_per_patient=IMAGES_PER_PATIENT, testing=False):
    if testing:
        test_images = datasets.ImageFolder(root=osP.join(path, "test"), transform=DATASET_TRANSFORM)
        return test_images, len(list(test_images)), None

    # create training datasets including lables with their respective class
    AD_files = FileLister(root=osP.join(path, "train", "AD"), 
                          masks="*.jpeg", recusive=False).map(label_file)
    NC_files = FileLister(root=osP.join(path, "train", "NC"), 
                          masks="*.jpeg", recusive=False).map(label_file)

    # batch data, grouped by patient ID 
    # buffer shuffle used to shuffle batches corresponding to patient within entire bucket
    AD_batch = AD_files.bucketbatch(use_in_batch_shuffle=False, 
                                    batch_size=imgs_per_patient, sort_key=sort_patients)
    NC_batch = NC_files.bucketbatch(use_in_batch_shuffle=False, 
                                    batch_size=imgs_per_patient, sort_key=sort_patients)
    
    # stratified split of AD & NC images to get validation data
    val_size = 0.2
    AD_train, AD_val = AD_batch.random_split(weights={
                                                "train": 0.8,
                                                "validation": 0.2},
                                                total_length=len(list(AD_batch)),
                                                seed=1)
    NC_train, NC_val = NC_batch.random_split(weights={
                                                "train": 0.8,
                                                "validation": 0.2},
                                                total_length=len(list(NC_batch)),
                                                seed=2)

    # combine AD & NC class splits
    # unbatch data 
    # shuffle data
    train_data = AD_train.concat(NC_train).unbatch().shuffle()
    val_data = AD_val.concat(NC_val).unbatch().shuffle()
    num_train_datapoints = len(list(train_data))

    # apply sharing filter to data 
    # open images and apply transforms to images
    train_images = train_data.sharding_filter().map(open_image).map(apply_tf)
    val_images = val_data.sharding_filter().map(open_image).map(apply_tf)

    return train_images, num_train_datapoints, val_images

"""
Main method to ensure multiprocessing doesn't break on Windows devices
"""
def main():
    pass

if __name__ == '__main__':
    main()