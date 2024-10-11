from modules import VisionTransformer
from dataset import *
import torch
from torchsummary import summary
import matplotlib.pyplot as plt


def train():
    NUM_EPOCH = 600
    LEARNING_RATE = 0.0002
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = VisionTransformer(6, 16, (3, 256, 256), 128, 8, 4, device).to(device)
    print(summary(model, (3, 256, 256)))
    valLoader = getValLoader()
    testLoader = getTestLoader()

    trainAccList = []
    trainLossList = []
    valAccList = []
    valLossList = []

    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    print("Beginning Training")
    for epoch in range(NUM_EPOCH):
        print(f'Epoch {epoch+1}/{NUM_EPOCH}:', end = ' ')
        trainLoader = getTrainLoader()
        model.train()
        trainAcc = 0
        trainLoss = 0
        for i, data in enumerate(trainLoader):
            imgs = data[0].to(device)
            labels = data[1].to(device)
            outputs = model(imgs)
            a, predLabels = torch.max(outputs.data, 1)
            trainAcc += (labels == predLabels).sum().item()
            loss = crit(outputs, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            trainLoss += loss.item()
        trainAcc /= len(trainLoader.dataset)
        trainLoss /= len(trainLoader)
        trainAccList.append(trainAcc)
        trainLossList.append(trainLoss) 
        print(f"Training Loss = {trainLoss}, Training Accuracy = {trainAcc}")  
        
        if ((epoch + 1) % 5 == 0):
            model.eval()
            valAcc = 0
            valLoss = 0
            torch.no_grad()
            for i, (imgs, trueLabels) in enumerate(valLoader):
                
                imgs = imgs.to(device)
                trueLabels = trueLabels.to(device)
                outputs = model(imgs)
                
                loss = crit(outputs, trueLabels)
            
                a, predLabels = torch.max(outputs.data, 1)
                
                valAcc += (trueLabels == predLabels).sum().item()
                valLoss += loss.item()

            valAcc /= len(valLoader.dataset)
            valLoss /= len(valLoader)
            valAccList.append(valAcc)
            valLossList.append(valLoss) 
            print(f"Validation Loss = {valLoss}, Validation Accuracy = {100 * valAcc} %")
        if ((epoch + 1) % 20 == 0):
            torch.save(model, f"model_epoch_{epoch + 1}")
            plt.plot(range(1, epoch + 2), trainAccList, label = "Train Accuracy")
            plt.savefig("trainAccuracyPlot.jpg")
            plt.close()
            plt.plot(range(1, epoch + 2), trainLossList, label = "Train Loss")
            plt.savefig("trainLossPlot.jpg")
            plt.close()
            plt.plot(range(5, epoch + 2, 5), valAccList, label = "Validation Accuracy")
            plt.savefig("valAccuracyPlot.jpg")
            plt.close()
            plt.plot(range(5, epoch + 2, 5), valLossList, label = "Validation Loss")
            plt.savefig("valLossPlot.jpg")
            plt.close()

if __name__ == '__main__':
    train()