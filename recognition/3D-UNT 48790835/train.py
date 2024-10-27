import torch
import numpy as np
import random
import argparse
from modules import UNet3D
from dataset import MRIDataset_pelvis
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt


## set random seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--lr',default=0.001)
parser.add_argument('--epoch',default=20)
parser.add_argument('--device',default='cuda')
parser.add_argument('--loss',default='dice')
parser.add_argument('--dataset_root', type=str, default=r'C:\Users\111\Desktop\3710\新建文件夹\数据集\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1', help='Root directory of the dataset')
args = parser.parse_args()

##define the model
model=UNet3D(in_channel=1, out_channel=6).cuda()

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        assert inputs.shape == targets.shape, f"Shapes don't match {inputs.shape} != {targets.shape}"
        inputs = inputs[:,1:]                                                       # skip background class
        targets = targets[:,1:]                                                     # skip background class
        axes = tuple(range(2, len(inputs.shape)))                                   # sum over elements per sample and per class
        intersection = torch.sum(inputs * targets, axes)
        addition = torch.sum(torch.square(inputs) + torch.square(targets), axes)
        return 1 - torch.mean((2 * intersection + self.smooth) / (addition + self.smooth))

##define the loss
if args.loss =='mse':
    criterion = torch.nn.MSELoss().cuda()
elif args.loss =='dice':
    criterion = DiceLoss().cuda()
elif args.loss =='ce':
    criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []

##define the train-dataloader and test_dataloader
train_dataset = MRIDataset_pelvis(mode='train',dataset_path=args.dataset_root)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_dataset = MRIDataset_pelvis(mode='test',dataset_path=args.dataset_root)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

##the training and valid process
start_time =time.time()
for epoch in range(args.epoch):
    model.train()
    train_epoch_loss = []

    for idx,(data_x,data_y) in enumerate(train_dataloader):
        data_x = data_x.to(torch.float32).cuda()
        data_y = data_y.to(torch.float32).cuda()
        labely=torch.nn.functional.one_hot(data_y.squeeze(1).long(),num_classes=6).permute(0,4,1,2,3).float()
        outputs = model(data_x)
        optimizer.zero_grad()
        loss = criterion(labely,outputs)
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())
        if idx%(len(train_dataloader)//2)==0:
            epoch_time=time.time()-start_time
            print("epoch={}/{},{}/{}of train, loss={} epoch time{}".format(
                epoch, args.epoch, idx, len(train_dataloader),loss.item(),epoch_time))
    train_epochs_loss.append(np.average(train_epoch_loss))
    epoch_time=time.time()-start_time
    print(f'epoch{epoch}:',train_epochs_loss)
    if epoch%1==0:
        model.eval()
        valid_epoch_loss = []
        for idx,(data_x,data_y) in enumerate(test_dataloader):
            data_x = data_x.to(torch.float32).to(args.device)
            data_y = data_y.to(torch.float32).to(args.device)
            labely=torch.nn.functional.one_hot(data_y.squeeze(1).long(),num_classes=6).permute(0,4,1,2,3).float()
            outputs = model(data_x)
            loss = criterion(outputs,labely)
            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())
        valid_epochs_loss.append(np.average(valid_epoch_loss))
        ##save the trained model
        torch.save(model.state_dict(),f'epoch_{epoch}_loss{args.loss}1.pth')

#plot the loss graph
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(train_loss[:])
plt.title(f"train_loss_{args.loss}")
plt.subplot(122)
plt.plot(train_epochs_loss[1:],'-o',label="train_loss")
plt.plot(valid_epochs_loss[1:],'-o',label="valid_loss")
plt.title("epochs_loss")
plt.legend()
plt.savefig(f"train_loss_{args.loss}1.png")