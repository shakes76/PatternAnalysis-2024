from modules import VisionTransformer, ConvolutionalVisionTransformer
from dataset import *
import torch
from torchsummary import summary
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler

MODEL_PATH = "./model15/model_epoch_266"

def predict(modelPath:str = MODEL_PATH, device: str = "cpu"):
    
    model = torch.load(modelPath).to(device)
    testLoader = getTestLoader(batchSize=128)

    pred = []
    true = []
    
    with torch.no_grad():
        model.eval()
        torch.no_grad()
        for i, (imgs, trueLabels) in enumerate(testLoader):
            imgs = imgs.to(device)
            trueLabels = trueLabels.to(device)
            outputs = model(imgs)
            a, predLabels = torch.max(outputs.data, 1)
            pred.append(predLabels)
            true.append(trueLabels)
    
    tAcc += (trueLabels == predLabels).sum().item()
    tAcc /= len(testLoader.dataset)
    print(f"Test Accuracy = {100 * tAcc} %")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predict(device)