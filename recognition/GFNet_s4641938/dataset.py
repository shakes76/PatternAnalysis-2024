from os.path import join
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import v2

"""
ADNI_DataLoader handles all data loaded to pass to the neural network
"""
class ADNI_DataLoader():
    rootDataPath: str
    imageSize: int
    
    __gotTrain : bool
    trainDataset : datasets.ImageFolder
    trainLoader : DataLoader
    
    __gotTest : bool
    testDataset : datasets.ImageFolder
    testLoader : DataLoader
    
    def __init__(self, rootData = "", imageSize = 240):
        self.rootDataPath = rootData
        self.imageSize = imageSize
        self.__gotTrain = False
        self.__gotTest = False

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
                transform = v2.Compose([
                    v2.CenterCrop(self.imageSize),
                    v2.ColorJitter(brightness = 0.4, contrast = 0.1),
                    #v2.RandomPosterize(bits = 3, p = 0.5),
                    v2.Grayscale(),
                    v2.ToTensor(),
                    v2.RandomErasing(p=0.5),
                    v2.GaussianNoise(0.1,0.05),
                    #v2.Normalize([0.0385], [0.0741]),
                    ])
                
                path_to_dataset = join(self.rootDataPath, 'train')
                self.trainDataset = datasets.ImageFolder(path_to_dataset, transform=transform)
                self.trainLoader = DataLoader(self.trainDataset, batch_size=64, shuffle=True)
                self.__gotTrain = True
            return self.trainLoader
            
        elif (data_type == "test"):
            if (not self.__gotTest):
                transform = v2.Compose(
                    [v2.CenterCrop(self.imageSize),
                    #v2.Grayscale(),
                    v2.ToTensor(),
                    #v2.Normalize([0.0385], [0.0741]),
                    ])
                path_to_dataset = join(self.rootDataPath, 'test')
                self.testDataset = datasets.ImageFolder(path_to_dataset, transform=transform)
                self.testLoader = DataLoader(self.testDataset, batch_size=64, shuffle=True)
                self.__gotTest = True
            return self.testLoader