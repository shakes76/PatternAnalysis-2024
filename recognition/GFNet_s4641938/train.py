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
    # Set constants as defaults
    IMAGESIZE = 180
    ROOTDATAPATH = "/home/groups/comp3710/ADNI/AD_NC"
    EPOCHS = 50
    lr = 0.0001
    
    # Handle optional arguments
    if len(args) >= 4:
        ROOTDATAPATH = args[3]
    if len(args) >= 3:
        EPOCHS = int(args[2])
    if len(args) >= 2:
        IMAGESIZE = int(args[1])

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Set seed to try to make replicable
    seed = 42
    torch.manual_seed(seed)

    # Get the data
    print("Loading ADNI brain data")
    full_dataHandler = ADNI_DataLoader(rootData=ROOTDATAPATH, imageSize=IMAGESIZE)
    train_dataloader = full_dataHandler.get_dataloader(data_type="train")
    test_dataloader = full_dataHandler.get_dataloader(data_type="test")

    # Create model
    print("Creating GFNet model")
    model = GFNet(img_size=IMAGESIZE, in_chans=1, patch_size=16, embed_dim=512, depth=18, num_classes=2, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.25)
    model.to(device)

    # Create optimizer, criterion and schedular
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)
    
    # Training loop
    print("Training start")
    
    # Save variables to track time & metrics
    time_s = time.time()
    accuracy = []
    max_accuracy = 0
    times_taken = []
    N_BATCHES = len(train_dataloader)
    N_IMAGES = len(train_dataloader.dataset)
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch} out of {EPOCHS}")
        model.train()
        total_loss = 0
        for batch, (images, targets) in enumerate(train_dataloader):
            images = images.to(device)
            targets = targets.to(device)

            # Prediction & Error
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Step schedular
            scheduler.step()
            
            # Print loss every 100 batches
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(images)
                print(f"loss: {loss:>7f}  [{current:>5d}/{N_IMAGES:>5d}]")

        # Save time taken since beginning training
        times_taken.append(time.time() - time_s)
        print(f"{int(times_taken[-1] // 3600)}h {int((times_taken[-1] % 3600) // 60)}m {int(times_taken[-1] % 60)}s taken for epoch {epoch}")
        
        # Check if accuracy for test data has improved
        acc, test_loss = getAccuracy(test_dataloader, model, device, criterion, -1)
        acc = acc.item()
        if acc > max_accuracy:
            max_accuracy = acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("New best model", end = " || ")
        accuracy.append(acc)
        
        # Print loss and accuracy for training/test data per epoch
        print("Loss: ", test_loss, "test", total_loss/N_BATCHES, "train")
        train_acc, _ = getAccuracy(train_dataloader, model, device, criterion, 10)
        print("Test Accuracy:", acc, "Train Accuracy:", train_acc.item())

if __name__ == "__main__":
    main(argv)