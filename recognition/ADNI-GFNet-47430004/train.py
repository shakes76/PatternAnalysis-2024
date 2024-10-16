"""
Got inspiration from engine.py file of the following github repo:
https://github.com/shakes76/GFNet
And my train/evaluate code from the brain GAN code.
"""

import math
import sys
import os
import torch
import time
import json
import datetime

import matplotlib.pyplot as plt
import torch.optim as optim

from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy
from timm.utils import NativeScaler, get_state_dict
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

import utils
from modules import GFNet
from dataset import get_dataloaders

# Parameters
output_dir = 'test/model/train'

# Hyperparameters
epochs = 2
start_epoch = 0
lr = 1e-6

project = "ADNI-GFNet"
group = "GFNet",

class defaultArgs(object):
    def __init__(self):
        self.sched = "cosine"

config = defaultArgs()

def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 400

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)
        loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)

        if device == 'cuda':
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 200, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    print("Training GFNet for ADNI brain data\n")
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

    optimizer = optim.Adam(model.parameters(), lr)
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