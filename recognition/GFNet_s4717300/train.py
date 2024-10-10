# “train.py" containing the source code for training, validating, testing and saving your model. The model
# should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
# sure to plot the losses and metrics during training
from dataset import GFNetDataloader
from modules import GFNet
import torch.optim.adam 
import torch
import time
from utils import bcolors
import csv


# Setting up Coda
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)

# hyper-parameters
learning_rate = 0.001
step_size = 5
gamma = 0.1

batches = 16
patch_size = 16
embed_dim = 768
depth = 12
ff_ratio = 4.0
epochs = 30

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


model = GFNet(img_size=image_size, patch_size=patch_size, in_chans=channels, num_classes=num_classes, embed_dim=embed_dim, depth=depth, ff_ratio=ff_ratio)
model.to(device)
model.train()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #type: ignore
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
total_step = len(training)

losses = []
acc_hist = []

# TODO: Move this to predict
def evaluate_model():
    # Test the model
    print("==Testing====================")
    start = time.time() #time generation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = (100 * correct / total)
        print('Test Accuracy: {} %'.format(accuracy))

    end = time.time()
    elapsed = end - start
    print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    model.train()
    print("=============================")
    return accuracy

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
            scheduler.step()
        
            if i % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}" .format(epoch+1, epochs, i+1, total_step, loss.item()))
                losses.append(loss.item())

        acc_hist.append(evaluate_model())
    torch.save(model.state_dict(), 'GFNET-{}.pth'.format(round(acc_hist[-1]), 4))
    # Save losses to a CSV file
    with open('losses.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(losses)

    # Save acc_hist to a CSV file
    with open('acc_hist.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(acc_hist)



print("==Training====================")
start = time.time() #time generation
train_model()
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
