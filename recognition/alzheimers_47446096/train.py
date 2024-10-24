from modules import VisionTransformer, ConvolutionalVisionTransformer
from dataset import *
import torch
from torchsummary import summary
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler

def train(device: str = "cpu"):
    NUM_EPOCH = 1000
    LEARNING_RATE = 0.0001  

    model = VisionTransformer(12, 14, (1, 224, 224), 128, 8, 4, device).to(device)
    #model = torch.load("./model15/model_epoch_266").to(device)
    print(summary(model, (1, 224, 224)))
    valLoader = getValLoader(gpu = True, batchSize=128)
    scaler = GradScaler()

    trainAccList = []
    trainLossList = []
    valAccList = []
    valLossList = []

    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=0.03)
    maxValAcc = 0
    #schL1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 100)
    print("Beginning Training")
    for epoch in range(NUM_EPOCH):
        trainLoader = getTrainLoader(gpu = True, batchSize=128, workers=2)
        print(f'Epoch {epoch+1}/{NUM_EPOCH}:', end = ' ')
        model.train()
        trainAcc = 0
        trainLoss = 0
        for i, data in enumerate(trainLoader):
            opt.zero_grad()
            
            imgs = data[0].to(device)
            labels = data[1].to(device)
            
            with torch.autocast(device_type = device, dtype = torch.float16):
                outputs = model(imgs)
                loss = crit(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            a, predLabels = torch.max(outputs.data, 1)
            trainAcc += (labels == predLabels).sum().item()            
            trainLoss += loss.item()
        #schL1.step()
        trainAcc /= len(trainLoader.dataset)
        trainLoss /= len(trainLoader)
        trainAccList.append(trainAcc)
        trainLossList.append(trainLoss) 
        print(f"Training Loss = {trainLoss}, Training Accuracy = {trainAcc}")  
        
        if ((epoch + 1) % 1 == 0):
            with torch.no_grad():
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
        
        if (valAccList[-1] > maxValAcc or valAccList[-1] >= 0.8):
            maxValAcc = valAccList[-1]
            torch.save(model, f"model_epoch_{epoch + 1}")
        
        if ((epoch + 1) % 20 == 0):
            plt.plot(range(1, epoch + 2), trainAccList, label = "Train Accuracy")
            plt.savefig("trainAccuracyPlot.jpg")
            plt.close()
            plt.plot(range(1, epoch + 2), trainLossList, label = "Train Loss")
            plt.savefig("trainLossPlot.jpg")
            plt.close()
            plt.plot(range(1, epoch + 2, 1), valAccList, label = "Validation Accuracy")
            plt.savefig("valAccuracyPlot.jpg")
            plt.close()
            plt.plot(range(1, epoch + 2, 1), valLossList, label = "Validation Loss")
            plt.savefig("valLossPlot.jpg")
            plt.close()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(device)