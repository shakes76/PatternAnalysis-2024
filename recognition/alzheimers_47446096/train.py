from modules import VisionTransformer
from dataset import *
import torch
from torchsummary import summary

NUM_EPOCH = 600
LEARNING_RATE = 0.0003
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = VisionTransformer(4, 16, (3, 256, 256), 128, 8, 3, device).to(device)
#print(summary(model, (3, 256, 256)))
valLoader = getValLoader()
testLoader = getTestLoader()
trainLossList = []

crit = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

print("Beginning Training")
for epoch in range(NUM_EPOCH):
    print(f'Epoch {epoch+1}/{NUM_EPOCH}:', end = ' ')
    trainLoss = 0
    trainLoader = getTrainLoader()
    model.train()
    for i, data in enumerate(trainLoader):
        imgs = data[0].to(device)
        labels = data[1].to(device)
        outputs = model(imgs)
        loss = crit(outputs, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        trainLoss += loss.item()
    
    trainLossList.append(trainLoss/len(trainLoader)) 
    print(f"Training loss = {trainLossList[-1]}")  
    
    if ((epoch + 1) % 5 == 0):
        model.eval()
        valAcc = 0
        torch.no_grad()
        for i, (imgs, trueLabels) in enumerate(valLoader):
            
            imgs = imgs.to(device)
            trueLabels = trueLabels.to(device)

            outputs = model(imgs)

            a, predLabels = torch.max(outputs.data, 1)
            valAcc += (trueLabels == predLabels).sum().item()
        print(f"Validation set accuracy = {100 * valAcc / len(valLoader.dataset)} %")
    if ((epoch + 1) % 20 == 0):
        torch.save(model, f"model_epoch_{epoch + 1}")
