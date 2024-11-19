from modules import VisionTransformer, ConvolutionalVisionTransformer
from dataset import *
import torch
from torchsummary import summary
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler
from sklearn.metrics import confusion_matrix, f1_score

MODEL_PATH = "relativePathToModel"

def predict(modelPath:str = MODEL_PATH, device: str = "cpu"):
    '''
    Calculates the accuracy, f1 score and confusion matrix of
    the model at modelPath
    '''
    print("Beginning testing")

    model = torch.load(modelPath, weights_only=False).to(device)
    testLoader = getTestLoader(batchSize=64)

    pred = torch.tensor([], device = device)
    true = torch.tensor([], device = device)
    
    with torch.no_grad():
        model.eval()
        torch.no_grad()
        for i, (imgs, trueLabels) in enumerate(testLoader):
            imgs = imgs.to(device)
            trueLabels = trueLabels.to(device)
            outputs = model(imgs)
            a, predLabels = torch.max(outputs.data, 1)
            pred = torch.cat((pred, predLabels), 0)
            true = torch.cat((true, trueLabels), 0)
    
    tAcc = (true == pred).sum().item()
    tAcc /= len(pred)
    
    print(f"Test Accuracy = {100 * tAcc} %")

    true = true.cpu()
    pred = pred.cpu()

    print(f"F1 Score: {f1_score(true, pred)}")

    print(f"Confusion Matrix:\n {confusion_matrix(true, pred, labels=[0, 1])}")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predict(device = device)