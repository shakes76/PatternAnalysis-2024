import torch
from dataset import ADNI_DataLoader
from modules import GFNet
from functools import partial
from sys import argv
import torch.nn as nn

def getAccuracy(test_dataloader, model, device, criterion, max_subset : int = -1):
    # Since evaluating the model, set model to evaluate and no gradient
    model.eval()
    with torch.no_grad():
        # Create metric counters
        total_correct = 0
        total_images = 0
        total_loss = 0
        
        # Iterate through each batch in given dataloader
        for batch, (images, targets) in enumerate(test_dataloader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Compute predictions
            outputs = model(images)
            
            
            # Calculate loss of batch
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Calculat accuracy of batch model, where the model's prediction is chosen as class with highest score
            most_likely = torch.max(outputs, dim=1).indices
            correct = most_likely == targets
            total_correct += correct.sum()
            
            # Count total images
            total_images += len(images)

            # If selected to only use a subset of the entire dataset of the dataloader end early.
            if max_subset != -1 and batch > max_subset:
                break
        return total_correct/total_images, total_loss/total_images

def main(args):
    """
    Constants
    """
    # Set constants as defaults
    IMAGESIZE = 180
    ROOTDATAPATH = "/home/groups/comp3710/ADNI/"
    MODELPATH = "./model.pth"
    GOAL_ACCURACY = 0.8
    
    # Handle optional arguments
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

    # Create model and load
    print("Creating GFNet model for testing")
    model = GFNet(img_size=IMAGESIZE, in_chans=1, patch_size=16, embed_dim=512, depth=18, num_classes=2, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.25)
    model.to(device)
    model.load_state_dict(torch.load(MODELPATH, map_location=device))
    print("Loaded model from model file")
    
    # Create criterion and get accuracy
    criterion = nn.CrossEntropyLoss()
    accuracy, _ = getAccuracy(test_dataloader, model, device, criterion, -1)
    print(f"Overall accuracy of the trained GFNet model is {accuracy}")
    
    # Print if accuracy reaches goal
    if accuracy > GOAL_ACCURACY:
        print(f"Model successfully performs better than goal accuracy of {GOAL_ACCURACY}")
    else:
        print(f"Model performs worse than goal accuracy of {GOAL_ACCURACY}")

if __name__ == "__main__":
    main(argv)