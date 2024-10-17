from os.path import join
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
ADNI_DataLoader handles all data loaded to pass to the neural network
"""
class ADNI_DataLoader():
    rootDataPath: str
    imageSize: int
    dataTransforms : transforms.Compose
    
    __gotTrain : bool
    trainDataset : datasets.ImageFolder
    trainLoader : DataLoader
    
    __gotTest : bool
    testDataset : datasets.ImageFolder
    testLoader : DataLoader
    
    def __init__(self, rootData = "", imageSize = 240):
        self.rootDataPath = rootData
        self.imageSize = imageSize
        self.dataTransforms = self.__create_transforms__()
        self.__gotTrain = False
        self.__gotTest = False
        
    def __create_transforms__(self):
        all_transforms = transforms.Compose(
                [transforms.Resize(self.imageSize),
                transforms.CenterCrop(self.imageSize),
                transforms.ToTensor(),
                transforms.Normalize([0.0385, 0.0385, 0.0385], [0.0741, 0.0741, 0.0741])])
        return all_transforms

    def __calc_mean_and_std(self):
        # Initialize sums
        mean, std = 0, 0
        total_images = 0
        # Iterate through the dataset
        for images, _ in self.trainDataset:
            # Reshape images to [C, H, W]
            images = images.view(images.size(0), images.size(1), -1)
            # Calculate mean and std for the batch
            mean += images.mean(dim=(1,2))  # Sum means for all images
            std += images.std(dim=(1,2))    # Sum stds for all images
            total_images += images.size(0)

        mean /= total_images
        std /= total_images

        return mean, std

    
    def get_dataloader(self, data_type: str):
        if (data_type == "train"):
            if (not self.__gotTrain):
                path_to_dataset = join(self.rootDataPath, 'train')
                self.trainDataset = datasets.ImageFolder(path_to_dataset, transform=self.dataTransforms)
                self.trainLoader = DataLoader(self.trainDataset, batch_size=64, shuffle=True)
                self.__gotTrain = True
            return self.trainLoader
            
        elif (data_type == "test"):
            if (not self.__gotTest):
                path_to_dataset = join(self.rootDataPath, 'test')
                self.testDataset = datasets.ImageFolder(path_to_dataset, transform=self.dataTransforms)
                self.testLoader = DataLoader(self.testDataset, batch_size=64, shuffle=True)
                self.__gotTest = True
            return self.testLoader