#the source code for training, validating, testing and saving your model.
# The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”.
# Make sure to plot the losses and metrics during training

#foreign libraries
import argparse
from tqdm import tqdm
from torchvision.utils import save_image
import torch .optim as optim
import torch
import torch.nn as nn

#local libraries
import modules
import dataset

learning_rate = 1e-4
epochs = 3
batch_size = 16
dim = (256,128)
train_im_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
train_mask_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train"
test_im_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"
test_mask_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test"
val_im_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"
val_mask_dir = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate"

def validate_UNet():
    '''
    function for validating the model.
    should only ever be called after the model has been trained on train/test data
    validation data must be split off apart from train/test data
    Parameters:
    
    
    return:
        validation metric
    '''
    return None

def train_UNet(loader, model, opt, loss, scaler, device):
    '''
    function for training the model.
    Parameters:


    return:
        updated model?
    '''
    loop = tqdm(loader)
    for batch, (data,targets) in enumerate(loop):
        data = data.to(device=device)
        target = targets.float().unsqueeze().to(device=device)

        with torch.amp.autocast():
            predict = model(data)
            loss = loss(predict, targets)
        
        opt.zero_grad()
        scaler.scale(loss).backwards()
        scaler.step(opt)
        scaler.update()

        loop.set_postfix(loss=loss.item())
    return None

def test_UNet():
    '''
    function for testing the model.
    Parameters:


    return:
        updated model?
    '''
    return None

def Save_UNet():
    '''
    function for saving the model????????????
    Parameters:


    return:
        huh?????????
    '''
    #torch.save(model.state_dict(), "Unet.pth")
    return None



def create_Unet(force_CPU = False):
    '''
    ideally, this should be the only function a person wishing to use this package will need to call
    Parameters:
        train (Dataset_3d): the segment of the data to be trained on
        test (Dataset_3d): the segment of the data to be tested on
        size (int): the N*N*N dimension of the train & test data
        force_CPU (bool): if true, allow the program to run on CPU, otherwise will abort if no GPU detected
        
    return:
        The trained Unet model
    '''
    device = modules.test_GPU_connection(force_CPU)
    model = modules.Unet2d(3,1).to(device)
    loss = modules.DiceLoss(4)
    opt = optim.Adam(model.parameters(), lr = learning_rate)

    train_load = dataset.Dataset_2d(train_im_dir, train_mask_dir)
    test_load = dataset.Dataset_2d(test_im_dir, test_mask_dir)
    val_load = dataset.Dataset_2d(val_im_dir, val_mask_dir)
    
    scaler = torch.amp.grad_scaler()
    for epoch in range(epochs):
        train_UNet(train_load, model, opt, loss, scaler)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("File_path", help="file path to the dataset to be ??????? modelled on?????")
    parser.add_argument("-fc","--force_CPU", help="set if you wish to allow usage of CPU, if GPU not available", action="store_true", default=False)
    #parser.add_argument("dimensionality", help="specify wether the data is a 2d or 2d segment", type=int, default=3)
    parser.add_argument("size", help="the N*N*N dimensions of the images in the dataset", type=int)
    args = parser.parse_args()

    
    #create_Unet(args.File_path, args.size, args.force_CPU)