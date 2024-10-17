import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ADNI_DataLoader
from modules import GFNet

def main():
    """
    Constants
    """
    IMAGESIZE = 240
    ROOTDATAPATH = "/home/groups/comp3710/ADNI/"
    EPOCHS = 100
    lr = 0.001

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set seed to make replicable
    seed = 42
    torch.manual_seed(seed)

    # Get the data
    print("Loading ADNI brain data")
    full_dataHandler = ADNI_DataLoader(rootData=ROOTDATAPATH, imageSize=IMAGESIZE)
    train_dataloader = full_dataHandler.get_dataloader(data_type="train")
    test_dataloader = full_dataHandler.get_dataloader(data_type="test")

    # Create model
    #Could try converting into binary classification (classification head becomes sigmoid?)
    print("Creating GFNet model")
    model = GFNet(img_size=IMAGESIZE, patch_size=16, embed_dim=384, depth=12, num_classes=2, mlp_ratio=4)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()


    N_IMAGES = len(train_dataloader.dataset)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch} out of {EPOCHS}")
        model.train()
        for batch, (images, targets) in enumerate(train_dataloader):
            images = images.to(device)
            targets = targets.to(device)

            # Prediction & Error
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 100 == 0:
                acc = getAccuracy(test_dataloader, model, device, 5)
                loss, current = loss.item(), (batch + 1) * len(images)
                print(f"loss: {loss:>7f} accuracy: {acc:>7f}  [{current:>5d}/{N_IMAGES:>5d}]")


def getAccuracy(test_dataloader, model, device, max_subset : int = -1):
    with torch.no_grad():
        total_correct = 0
        total_images = 0
        for batch, (images, targets) in enumerate(test_dataloader):
            images.to(device)
            targets.to(device)
            outputs = model(images)
            total_correct += outputs == targets
            total_images += len(images)
            if max_subset != -1 and batch > max_subset:
                break

    return total_correct/total_images

if __name__ == "__main__":
    main()