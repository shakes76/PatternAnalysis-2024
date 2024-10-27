import torch
from dataset import ADNI_DataLoader
from modules import GFNet
from functools import partial
from sys import argv
import torch.nn as nn

def getAccuracy(test_dataloader, model, device, criterion, max_subset : int = -1):
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_images = 0
        total_loss = 0
        for batch, (images, targets) in enumerate(test_dataloader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            most_likely = torch.max(outputs, dim=1).indices #get class with highest score
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            correct = most_likely == targets
            total_correct += correct.sum()
            total_images += len(images)

            #print(f"[{batch}/{len(test_dataloader)}] Ongoing accuracy: {total_correct/total_images}")
            if max_subset != -1 and batch > max_subset:
                break
        return total_correct

def main(args):
    """
    Constants
    """
    IMAGESIZE = 180
    ROOTDATAPATH = "/home/groups/comp3710/ADNI/"
    MODELPATH = "./model.pth"
    GOAL_ACCURACY = 0.8 # Goal accuracy to achieve
    
    if len(args) >= 4:
        ROOTDATAPATH = args[3]
    if len(args) >= 3:
        MODELPATH = args[2] 
    if len(args) >= 2:
        IMAGESIZE = int(args[1])

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get the data
    print("Loading ADNI brain data")
    full_dataHandler = ADNI_DataLoader(rootData=ROOTDATAPATH, imageSize=IMAGESIZE)
    test_dataloader = full_dataHandler.get_dataloader(data_type="test")

    # Create model
    print("Creating GFNet model for testing")
    model = GFNet(img_size=IMAGESIZE, in_chans=1, patch_size=16, embed_dim=512, depth=18, num_classes=2, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.25)
    model.to(device)

    model.load_state_dict(torch.load(MODELPATH, map_location=device))
    print("Loaded model from model file")
    criterion = nn.CrossEntropyLoss()
    accuracy = getAccuracy(test_dataloader, model, device, criterion, -1)
    print(f"Overall accuracy of the trained GFNet model is {accuracy}")
    
    if accuracy > GOAL_ACCURACY:
        print(f"Model successfully performs better than goal accuracy of {GOAL_ACCURACY}")
    else:
        print(f"Model performs worse than goal accuracy of {GOAL_ACCURACY}")

if __name__ == "__main__":
    main(argv)