import dataset
import modules
import train
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# Add your own paths here
testImagesPath = "isic_data/ISIC2018_Task1-2_Test_Input"
testLabelsPath = "isic_data/ISIC2018_Task1_Test_GroundTruth"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    testDataSet = dataset.ISIC2017DataSet(testImagesPath, testLabelsPath, dataset.ISIC_transform_img(), dataset.ISIC_transform_label())
    testDataloader = DataLoader(testDataSet, batch_size=train.batchSize, shuffle=False)

    model = modules.Improved2DUnet()
    model.load_state_dict(torch.load(train.modelPath))
    model.to(device)
    print("Model Successfully Loaded")
    
    test(testDataloader, model, device)

def test(dataLoader, model, device):
    losses_validation = list()
    dice_similarities_validation = list()

    print("> Test Inference Commenced")
    start = time.time()
    model.eval()
    with torch.no_grad():
        print(dataLoader)
        for step, (images, labels) in enumerate(dataLoader):
            print(step)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            losses_validation.append(train.dice_loss(outputs, labels))
            dice_similarities_validation.append(train.dice_coefficient(outputs, labels))

            if (step == 0):
                train.save_segments(images, labels, outputs, 9, test=True)

        print('Test Loss: {:.5f}, Test Average Dice Similarity: {:.5f}'.format(train.get_average(losses_validation) ,train.get_average(dice_similarities_validation)))
    end = time.time()
    elapsed = end - start
    print("Test Inference took " + str(elapsed/60) + " mins in total")

if __name__ == "__main__":
    main()
