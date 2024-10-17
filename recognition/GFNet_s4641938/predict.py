import torch
from dataset import ADNI_DataLoader
from modules import GFNet
from functools import partial
from predict import getAccuracy

def getAccuracy(test_dataloader, model, device, max_subset : int = -1):
    with torch.no_grad():
        total_correct = 0
        total_images = 0
        for batch, (images, targets) in enumerate(test_dataloader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            most_likely = torch.max(outputs, dim=1).indices #get class with highest score
            correct = most_likely == targets
            total_correct += correct.sum()
            total_images += len(images)

            #print(f"[{batch}/{len(test_dataloader)}] Batch accuracy: {correct.sum()/len(images)}")
            if max_subset != -1 and batch > max_subset:
                break
        return total_correct/total_images


def main():
    """
    Constants
    """
    IMAGESIZE = 240
    ROOTDATAPATH = "/home/groups/comp3710/ADNI/"
    MODELPATH = "./model.pth"
    GOAL_ACCURACY = 0.8 # Goal accuracy to achieve
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get the data
    print("Loading ADNI brain data")
    full_dataHandler = ADNI_DataLoader(rootData=ROOTDATAPATH, imageSize=IMAGESIZE)
    test_dataloader = full_dataHandler.get_dataloader(data_type="test")

    # Create model
    print("Creating GFNet model for testing")
    model = GFNet(img_size=IMAGESIZE, patch_size=16, embed_dim=384, depth=12, num_classes=2, mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))
    model.to(device)

    model.load_state_dict(torch.load(MODELPATH, map_location=device))
    print("Loaded model from model file")
    accuracy = getAccuracy(test_dataloader, model, device, -1)
    print(f"Overall accuracy of the trained GFNet model is {accuracy}")
    
    if accuracy > GOAL_ACCURACY:
        print(f"Model successfully performs better than goal accuracy of {GOAL_ACCURACY}")
    else:
        print(f"Model performs worse than goal accuracy of {GOAL_ACCURACY}")

if __name__ == "__main__":
    main()