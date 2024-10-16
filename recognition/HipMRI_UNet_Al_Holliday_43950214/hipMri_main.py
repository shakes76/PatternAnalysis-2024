#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main driver script

@author: al
"""

import torch

import modules
#from predict import predict
from train import train

def main():
    chan = 1
    outDim = 64
    dev = torch.device("cuda")
    net = modules.UNet(chan, outDim, segDim = 6)
    
    # perform training
    net = train(net, dev, chan, outDim, numEpochs = 16)
    
    # perform validation
    # TODO : ^^^^^^^^^^^ DO THAT!!!!


if __name__ == "__main__":
    main()