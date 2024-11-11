import torch
import torch.nn as nn
import torch.nn.functional as F 
import time

from dataset import *
from modules import *


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("cuda")
if not torch.cuda.is_available():
    print("cpu")  
    
#hyperparameters
epochs = 10
learning_rate=0.001
image_size = 416
batch_size = 10

#Train data - change directories as needed
mask_dir = '/Users/mariam/Downloads/COMP3710_YOLO/ISIC-2017_Training_Part1_GroundTruth/'
image_dir = '/Users/mariam/Downloads/COMP3710_YOLO/ISIC-2017_Training_Data/'
labels = '/Users/mariam/Downloads/COMP3710_YOLO/ISIC-2017_Training_Part3_GroundTruth.csv'
train_dataset = ISICDataset(image_dir, mask_dir, labels, image_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#Model
model = YOLO(2)
model.to(device)
checkpoint_path = "model.pt"

#optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
criterion = YOLO_loss()

#learning rate schedule, using because SGD is dumb, adam has its own learning rate
total_step = len(train_dataloader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=learning_rate,
                                                steps_per_epoch=total_step, epochs=epochs)

#Train
model.train()
start = time.time()
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        #Forward pass
        outputs = model(images)
        total_loss = 0
        for a in range(batch_size):
            loss = criterion(outputs[a], labels[a])
            total_loss += loss

        #Backwards and optimize
        optimizer.zero_grad()
        total_loss.requires_grad = True
        total_loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            print("Epoch [{}/{}], Step[{},{}] Loss: {:.5f}".format(epoch+1, epochs, i+1, total_step, total_loss.item()))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
                }, checkpoint_path)

        scheduler.step()
end = time.time()
elapsed = end - start
print("Training took {} secs or {} mins.".format(elapsed, elapsed/60))

#Test data
mask_dir = '/Users/mariam/Downloads/COMP3710_YOLO/ISIC-2017_Test_v2_Part1_GroundTruth/'
image_dir = '/Users/mariam/Downloads/COMP3710_YOLO/ISIC-2017_Test_v2_Data/'
labels = '/Users/mariam/Downloads/COMP3710_YOLO/ISIC-2017_Test_v2_Part3_GroundTruth.csv'
test_dataset = ISICDataset(image_dir, mask_dir, labels, 416)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

#Test
model.eval()
torch.set_grad_enabled(True)
start = time.time()
total = 0
total_step = len(test_dataloader)

for i, (images, labels) in enumerate(test_dataloader):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)

    #Calculate IoU
    for a in range(batch_size):
        best_box = filter_boxes(outputs[a])
        if best_box is not None:
            best_box = torch.reshape(best_box, (1, 7))
            iou = single_iou(best_box, labels[a,:])
            total += iou[0]

    #Keep track of average
    average = total/(i+1)

    if (i+1) % 50 == 0:
        print("Step[{},{}] IoU average: {:.5f}".format(i+1, total_step, average))

end = time.time()
elapsed = end - start
print("Testing took {} secs or {} mins.".format(elapsed, elapsed/60))