"""
Got inspiration from main_gfnet.py file of the following github repo:
https://github.com/shakes76/GFNet
"""
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict

import utils
import os

import torch.optim as optim
from train import train_one_epoch, evaluate, defaultArgs
from dataset import get_dataloaders
from timm.utils import NativeScaler

from modules import GFNet

# Hyperparameters
hyperparameter_1 = None
epochs = 5
start_epoch = 0
lr = 1e-6
output_dir = 'test/model/'

config = defaultArgs()

project = "ADNI-GFNet"
group = "GFNet",
log_config = {
        "id": 0,
        "machine": "a100",
        "architecture": "gfnet-xs",
        "model": "GFNet",
        "dataset": "ADNI",
        "epochs": 300,
        "optimizer": "adam",
        "loss": "crossentropy",
        "metric": "accuracy",
        #~ "dim": 64,
        "depth": 12,
        "embed_dim": 384,
        "batch_size": 128
}

# Code from main_gfnet.py will go here
def main():
    print("Main of Modules - modules compiles/runs\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    criterion = torch.nn.CrossEntropyLoss()
    train_loader, test_loader = get_dataloaders(None)
    epoch = 0
    loss_scaler = NativeScaler()

    model = GFNet(num_classes=2, in_chans=1)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler, _ = create_scheduler(config, optimizer)

    print(f"Start training for {epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(start_epoch, epochs):
        
        train_stats = train_one_epoch(
            model, criterion, train_loader,
            optimizer, device, epoch, loss_scaler)

        lr_scheduler.step(epoch)
        
        if (epoch + 1) % 20 == 0:
            file_name = 'checkpoint_epoch%d.pth' % epoch
            checkpoint_path = os.path.join(output_dir, file_name)
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
            }, checkpoint_path)

        test_stats = evaluate(test_loader, model, device)
        print(f"Accuracy of the network on the {len(test_loader)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if max_accuracy == test_stats["acc1"]:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
            utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                }, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if output_dir and utils.is_main_process():
            with open(output_dir + "log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    main()