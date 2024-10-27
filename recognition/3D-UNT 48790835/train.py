# source code for training, validating, testing and saving the model
import dataset
import modules
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
import numpy as np

# Hyper-parameters
num_epochs = 30
learning_rate = 5 * 10**-4
batchSize = 16
learning_rate_decay = 0.985

# set up the funcitonality from the imported dataset.py and modules.py

# Add your own paths here
validationImagesPath = "isic_data/ISIC2018_Task1-2_Validation_Input"
trainImagesPath = "isic_data/ISIC2018_Task1-2_Training_Input_x2"
validationLabelsPath = "isic_data/ISIC2018_Task1_Validation_GroundTruth"
trainLabelsPath = "isic_data/ISIC2018_Task1_Training_GroundTruth_x2"
modelPath = "model.pt"

def init():
    validDataSet = dataset.ISIC2018DataSet(validationImagesPath, validationLabelsPath, dataset.img_transform(), dataset.label_transform())
    validDataloader = DataLoader(validDataSet, batch_size=batchSize, shuffle=False)
    trainDataSet = dataset.ISIC2018DataSet(trainImagesPath, trainLabelsPath, dataset.img_transform(), dataset.label_transform())
    trainDataloader = DataLoader(trainDataSet, batch_size=batchSize, shuffle=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    dataLoaders = dict()
    dataLoaders["valid"] = validDataloader
    dataLoaders["train"] = trainDataloader

    dataSets = dict()
    dataSets["valid"] = validDataSet
    dataSets["train"] = trainDataSet

    return dataSets, dataLoaders, device

def main():
    dataSets, dataLoaders, device = init()
    model = modules.Improved2DUnet()
    model = model.to(device)

    # training and validating
    train_and_validate(dataLoaders, model, device)

    # saving
    torch.save(model.state_dict(), modelPath)

def train_and_validate(dataLoaders, model, device):
    # Define optimization parameters and loss according to Improved Unet Paper.
    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=10**-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=learning_rate_decay)

    losses_training = list()
    dice_similarities_training = list()
    losses_valid = list()
    dice_similarities_valid = list()

    print("Training and Validation Commenced:")
    start = time.time()
    epochNumber = 0

    for epoch in range(num_epochs):
        epochNumber += 1
        train_loss, train_coeff = train(dataLoaders["train"], model, device, criterion, optimizer, scheduler)
        valid_loss, valid_coeff = validate(dataLoaders["valid"], model, device, criterion, epochNumber)

        losses_training.append(train_loss)
        dice_similarities_training.append(train_coeff)
        losses_valid.append(valid_loss)
        dice_similarities_valid.append(valid_coeff)


        print ("Epoch [{}/{}], Training Loss: {:.5f}, Training Dice Similarity {:.5f}".format(epoch+1, num_epochs, losses_training[-1], dice_similarities_training[-1]))
        print('Validation Loss: {:.5f}, Validation Average Dice Similarity: {:.5f}'.format(get_average(losses_valid) ,get_average(dice_similarities_valid)))
        
    
    end = time.time()
    elapsed = end - start
    print("Training & Validation Took " + str(elapsed/60) + " Minutes")

    save_list_as_plot(trainList=losses_training, valList=losses_valid, type="Loss", path="LossCurve.png")
    save_list_as_plot(trainList=dice_similarities_training, valList=dice_similarities_valid, type="Dice Coefficient", path="DiceCurve.png")


def train(dataLoader, model, device, criterion, optimizer, scheduler):
    model.train()

    losses = list()
    coefficients = list()

    for images, labels in dataLoader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        coefficients.append(dice_coefficient(outputs, labels).item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    scheduler.step()

    return get_average(losses), get_average(coefficients)

def validate(dataLoader, model, device, criterion, epochNumber):

    losses = list()
    coefficients = list()

    model.eval()
    with torch.no_grad():
        for step, (images, labels) in enumerate(dataLoader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            coefficients.append(dice_coefficient(outputs, labels).item())

            if (step == 0):
                save_segments(images, labels, outputs, 9, epochNumber)
    
    return get_average(losses), get_average(coefficients)


# Variable numList must be a list of number types only
def get_average(numList):
    
    size = len(numList)
    count = 0
    for num in numList:
        count += num
    
    return count / size

def dice_loss(outputs, labels):
    
    return 1 - dice_coefficient(outputs, labels)

def dice_coefficient(outputs, labels, epsilon=10**-8):

    intersection = (outputs * labels).sum()
    denom = (outputs + labels).sum() + epsilon
    diceCoefficient = (2. * intersection) / denom
    return diceCoefficient

def print_model_info(model):

    print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
    print(model)

def save_segments(images, labels, outputs, numComparisons, epochNumber=num_epochs, test=False):

    if numComparisons > batchSize:
        numComparisons = batchSize
    
    images=images.cpu()
    labels=labels.cpu()
    outputs=outputs.cpu()

    fig, axs = plt.subplots(numComparisons, 3)
    axs[0][0].set_title("Image")
    axs[0][1].set_title("Ground Truth")
    axs[0][2].set_title("Predicted")
    for row in range(numComparisons):
        img = images[row]
        img = img.permute(1,2,0).numpy()
        label = labels[row]
        label = label.permute(1,2,0).numpy()
        pred = outputs[row]
        pred = pred.permute(1,2,0).numpy()
        axs[row][0].imshow(img)
        axs[row][0].xaxis.set_visible(False)
        axs[row][0].yaxis.set_visible(False)

        axs[row][1].imshow(label, cmap="gray")
        axs[row][1].xaxis.set_visible(False)
        axs[row][1].yaxis.set_visible(False)

        axs[row][2].imshow(pred, cmap="gray")
        axs[row][2].xaxis.set_visible(False)
        axs[row][2].yaxis.set_visible(False)
    
    if (not test):
        fig.suptitle("Validation Segments Epoch: " + str(epochNumber))
        #fig.tight_layout()
        plt.savefig("ValidationSegmentsEpoch" + str(epochNumber))
    else:
        fig.suptitle("Test Segments")
        #fig.tight_layout()
        plt.savefig("TestSegments")
    plt.close()

def save_list_as_plot(trainList, valList, type, path):

    if (len(trainList) != len(valList)):
        print("ERROR: Cannot display!")
    
    length = len(trainList)
    xList = list()
    x = 1
    for i in range(length):
        xList.append(x)
        x += 1

    plt.xticks(np.arange(min(xList), max(xList)+1, 1.0))
    plt.plot(xList, trainList, label="Training " + type)
    plt.plot(xList, valList, label="Validation " + type)
    plt.legend()
    plt.title("Training and Validation " + type + " Over Epochs")
    plt.savefig(fname=path)
    plt.close()





if __name__ == "__main__":
    main()
