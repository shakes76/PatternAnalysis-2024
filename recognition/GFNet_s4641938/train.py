import torch
import torch.nn as nn
import torch.optim as optim
from dataset import ADNI_DataLoader
from modules import GFNet
import time
from functools import partial
from predict import getAccuracy
from sys import argv

def main(args):
    """
    Constants
    """
    IMAGESIZE = 240
    ROOTDATAPATH = "/home/groups/comp3710/ADNI/AD_NC"
    EPOCHS = 100
    lr = 0.001
    
    if len(args) >= 4:
        ROOTDATAPATH = args[3]
    if len(args) >= 3:
        EPOCHS = int(args[2])
    if len(args) >= 2:
        IMAGESIZE = int(args[1])

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
    model = GFNet(img_size=IMAGESIZE, patch_size=16, embed_dim=384, depth=12, num_classes=2, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1, dropcls=0.25)
    #model = GFNetPyramid(
    #        img_size=IMAGESIZE,
    #        patch_size=4, embed_dim=[96, 192, 384, 768], depth=[3, 3, 18, 3],
    #        mlp_ratio=[4, 4, 4, 4],
    #        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.4, init_values=1e-6
    #    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)
    
    print("Training start")
    time_s = time.time()
    accuracy = []
    max_accuracy = 0
    times_taken = []
    prev_time = 0
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
            
            #scheduler.step()
            #print(getAccuracy(test_dataloader, model, device, 1).item())
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(images)
                print(f"loss: {loss:>7f}  [{current:>5d}/{N_IMAGES:>5d}]")

        times_taken.append(time.time() - time_s)
        prev_time = times_taken[-1]
        print(f"{int(times_taken[-1] // 3600)}h {int((times_taken[-1] % 3600) // 60)}m {int(times_taken[-1] % 60)}s taken for epoch {epoch}")
        acc = getAccuracy(test_dataloader, model, device, -1).item()
        if acc > max_accuracy:
            max_accuracy = acc
            torch.save(model.state_dict(), 'model.pth')
            print("New best model", end = " || ")
        accuracy.append(acc)
        print("Accuracy:", acc)

if __name__ == "__main__":
    main(argv)