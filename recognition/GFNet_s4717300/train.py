# “train.py" containing the source code for training, validating, testing and saving your model. The model
# should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
# sure to plot the losses and metrics during training
from dataset import GFNetDataloader
from modules import GFNet
import torch.optim.adamw
import torch
import time
from utils import bcolors
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import csv
import argparse

parser = argparse.ArgumentParser(description='Optional argument example')

# Define an optional argument, e.g., '--myarg'
parser.add_argument('--mpath', type=str, help='Model weights to load')
parser.add_argument('--tag', type=str, help='tag to label the models')

# Parse the arguments
args = parser.parse_args()

loaded_model = args.mpath
if not args.tag:
    print('No tag supplied')
    exit(3)
tag = args.tag

# Setting up Coda
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)

# hyper-parameters
learning_rate = 0.001
weight_decay = 0.1
dropout = 0.3
drop_path = 0.3

batches = 24
patch_size = 8
embed_dim = 384
depth = 8
ff_ratio = 2
epochs = 30
warmup_epochs = 5

## Load data
gfDataloader = GFNetDataloader(batches)
gfDataloader.load()
training, test = gfDataloader.get_data()
image_size = gfDataloader.get_meta()['img_size']
if not training or not test:
    print("Problem loading data, please check dataset is commpatable with dataloader including all hyprparameters")
    exit(1)


# Image Info
channels = 1
num_classes = 2
img_shape = (channels, image_size, image_size)


model = GFNet(img_size=image_size, patch_size=patch_size, in_chans=channels, num_classes=num_classes, embed_dim=embed_dim, depth=depth, ff_ratio=ff_ratio, dropout=dropout, drop_path_rate=drop_path)
model.to(device)
model.train()
if loaded_model:
    model.load_state_dict(torch.load(loaded_model, weights_only=True))

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
total_step = len(training)
cosine_t_max = epochs - warmup_epochs
warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_t_max)

scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

losses = []
val_losses = []

# TODO: Move this to predict
def evaluate_model(final=False):
    # Test the model
    print("==Testing====================")
    start = time.time() #time generation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        avg_loss = 0 
        count = 0
        for images, labels in test:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            count += 1
            correct += (predicted == labels).sum().item()
            avg_loss += criterion(outputs, labels) 
            if not final and count > 10:
                break
        val_losses.append(avg_loss / count)

        accuracy = (100 * correct / total)
        print('Test Accuracy: {} %'.format(accuracy))

    end = time.time()
    elapsed = end - start
    print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    model.train()
    print("=============================")
    return accuracy, (avg_loss / count)

def train_model():
    ## Start training loop
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(training, 0):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if i % 1 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}" .format(epoch+1, epochs, i+1, total_step, loss.item()))


        losses.append(loss.item())
        result, v_loss = evaluate_model() 
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'GFNET-e{}-{}-{}.pth'.format(epoch, round(result, 4), tag))

            if abs(v_loss - loss.item()) > 0.35:
                print('=======OVERFITTED-TERMINATED======')
                exit(2)

        scheduler.step() 
    result, _ = evaluate_model(final=True) 
    # torch.save(model.state_dict(), 'GFNET-{}.pth'.format(round(result, 4)))
    torch.save(model.state_dict(), 'GFNET-{}-{}.pth'.format(round(result, 4), tag))
    # Save losses to a CSV file
    with open('losses.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(losses)

    # Save acc_hist to a CSV file
    with open('val_loss.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(val_losses)



print("==Training====================")
start = time.time() #time generation
train_model()
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
