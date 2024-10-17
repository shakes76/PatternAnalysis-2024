import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ADNI_DataLoader
from modules import GFNet
import time
from functools import partial
from predict import getAccuracy

def main():
    """
    Constants
    """
    IMAGESIZE = 240
    ROOTDATAPATH = "/home/groups/comp3710/ADNI/AD_NC"
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
    print("Creating GFNet model")
    model = GFNet(img_size=IMAGESIZE, patch_size=16, embed_dim=384, depth=12, num_classes=2, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print("Training start")
    time_s = time.time()
    accuracy = []
    max_accuracy = 0
    times_taken = []
    N_BATCHES = len(train_dataloader)
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
                loss, current = loss.item(), (batch + 1) * len(images)
                print(f"loss: {loss:>7f}  [{current:>5d}/{N_IMAGES:>5d}]")

        times_taken.append(time.time() - time_s)
        print(f"{int(times_taken[-1] // 3600)}h {int((times_taken[-1] % 3600) // 60)}m {int(times_taken[-1] % 60)}s taken for epoch {epoch}")
        acc = getAccuracy(test_dataloader, model, device, -1).item()
        if acc > max_accuracy:
            max_accuracy = acc
            torch.save(model.state_dict(), 'model.pth')
            print("New best model", end = " || ")
        accuracy.append(acc)
        print("Accuracy:", acc)

if __name__ == "__main__":
    main()