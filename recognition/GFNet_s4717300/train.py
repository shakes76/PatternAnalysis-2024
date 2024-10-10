# “train.py" containing the source code for training, validating, testing and saving your model. The model
# should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
# sure to plot the losses and metrics during training
from dataset import GFNetDataloader
from modules import GFNet
import torch.optim.adam 
import torch
import time
from utils import bcolors


# Setting up Cuda
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)

# hyper-parameters
channels = 1
image_size = 28
img_shape = (channels, image_size, image_size)
learning_rate = 0.001
batches = 16

# Epochs
epochs = 1






## Load data
gfDataloader = GFNetDataloader(batches)
gfDataloader.load()
training, test = gfDataloader.get_data()
if not training or not test:
    print(bcolors.FAIL + "Problem loading data, please check dataset is commpatable with dataloader inclusing all hyprparameters" + bcolors.ENDC )
    exit(1)

model = GFNet(img_size=image_size, patch_size=2, in_chans=channels, num_classes=10, embed_dim=784, depth=12, ff_ratio=4.)
model.to(device)
model.train()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #type: ignore
total_step = len(training)

losses = []

print(bcolors.OKGREEN + "==Training====================" + bcolors.ENDC)
start = time.time() #time generation
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

        print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}" .format(epoch+1, epochs, i+1, total_step, loss.item()))



end = time.time()
elapsed = end - start
print(bcolors.OKGREEN + "Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total"  + bcolors.ENDC)


## Start training loop
