from modules import VisionTransformer
from dataset import *
import torch

NUM_EPOCH = 5
LEARNING_RATE = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = VisionTransformer(128, 16, (3, 256, 256), 200, 4, device).to(device)
trainLoader = getTrainLoader()
testLoader = getTestLoader()
trainLossList = []

crit = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

for epoch in range(NUM_EPOCH):
    print(f'Epoch {epoch+1}/{NUM_EPOCH}:', end = ' ')
    trainLoss = 0

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
    

model.eval()
testAcc = 0
torch.no_grad()
for i, (imgs, trueLabels) in enumerate(testLoader):
    
    imgs = imgs.to(device)
    trueLabels = trueLabels.to(device)

    outputs = model(imgs)

    a, predLabels = torch.max(outputs.data, 1)
    testAcc += (trueLabels == predLabels).sum().item()

print(f"Test set accuracy = {100 * testAcc / (len(testLoader) * testLoader.batch_size)} %")