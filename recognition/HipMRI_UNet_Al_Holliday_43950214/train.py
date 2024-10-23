# -*- coding: utf-8 -*-
"""
Training script. Initially I'm just adapting Shakes' training code from the CNN example for
MNIST: https://colab.research.google.com/drive/1K2kiAJSCa6IiahKxfAIv4SQ4BFq7YDYO?usp=sharing

Now refactored into a single function so it can be used in the main driver script.

@author: al
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import dataset
import modules
from dice import dice_coeff


def train(net, dev,channels = 1, outDimension = 64, numEpochs = 8):
    """
    The training function. Performs the training sequence (and a little bit of testing)
    
    Parameters:
        net: the model to train
        dev: the device to put the tensors on
        channels: the number of input channels to use
        outDimension: the number of filters the output should have
        numEpochs: the number of epochs to train
        
    Returns: the trained model
    """
    chan = channels
    outDim = outDimension

    epochs = numEpochs
    #dev = torch.device("cuda")
    trans = transforms.Resize((256, 256))
    # example HipMRI dataset root folders (feel free to replace with your own):
    # For Woomy (my laptop)
    #hipMriRoot = "C:\\Users\\al\\HipMRI"
    # For the lab computers
    #hipMriRoot = "H:\\HipMRI"
    # For Rangpur
    hipMriRoot = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    hipmri2dtrain = dataset.HipMRI2d(hipMriRoot, imgSet = "train", transform = trans, applyTrans = True)
    hipmri2dtest = dataset.HipMRI2d(hipMriRoot, imgSet = "test", transform = trans, applyTrans = True)
    trainLoader = DataLoader(hipmri2dtrain, batch_size = 8, shuffle = False)
    testLoader = DataLoader(hipmri2dtest, batch_size = 8, shuffle = False)

    #net = modules.UNet(chan, outDim, segDim = 6)
    #if not next(net.parameters()).device == dev:
    net = net.to(dev)

    lossFunc = nn.CrossEntropyLoss()
    optm = torch.optim.SGD(net.parameters())
    
    print("Training")
    net.train()
    for epoch in range(epochs):
        for i, (img, seg) in enumerate(trainLoader):
            #print(i)
            img = img.to(dev)
            #seg = nn.functional.one_hot(seg.long())
            #seg = seg.squeeze().long()
            seg = seg.squeeze()
            seg = seg.to(dev)
            
            out = net(img)
            #print(out.shape)
            # This is no longer necessary!
            #seg = TF.center_crop(seg, output_size = out.size(2))
            loss = lossFunc(out, seg)
            optm.zero_grad()
            loss.backward()
            optm.step()
            if (i+1) % 100 == 0:
                print ("Epoch [{}/{}], Loss: {:.5f}"
                        .format(epoch+1, epochs, loss.item()))

    print("Done!")
    print("Testing")
    # save the weights
    torch.save(net.state_dict(), "./weights.pth")
    # perform initial testing (i.e. NOT validation) here
    diceLosses = []
    net.eval()
    with torch.no_grad():
        for img, seg in testLoader:
            img = img.to(dev)
            #seg = seg.squeeze().long()
            #seg = seg.squeeze()
            seg = nn.functional.one_hot(seg.squeeze(), num_classes = 6)
            seg = torch.permute(seg, (0, 3, 1, 2))
            seg = seg.to(dev)
            out = net(img)
            #out = torch.permute(out, (0, 2, 3, 1)) # put the chan dim last
            #out = torch.argmax(out, dim = -1)
            #out = out[:, None, :, :] # reshape back to (batch, chan, h, w)
            diceSimilarity = dice_coeff(out, seg, dev, 6)
            print("current dice: {:.5f}".format(diceSimilarity.cpu().item()))
            diceLosses.append(diceSimilarity.cpu().item())

    print("Done!")
    avgDice = sum(diceLosses) / len(diceLosses)
    print("average dice from initial testing: {:.5f}".format(avgDice))
    
    return net



# one option is to upsize the output to the size of the original.
# this probably won't work well as it's just blowing up a smaller image
# and won't line up exactly with the target segment map.
#out = trans(out)
# Another option is to crop the target. This will probably work better
#seg = TF.center_crop(seg, output_size = out.size(1))
#loss = lossFunc(out[0], seg[0]) # probably will not work
    
