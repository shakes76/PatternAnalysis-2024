from torchvision import transforms

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
### basic normalisation for RGB inesity values per channel - use 0.5 to place intensity values between [-1, 1]
DATASET_TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)),
    transforms.ToTensor()
    ])