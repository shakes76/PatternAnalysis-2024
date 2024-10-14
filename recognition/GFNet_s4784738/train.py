"""
Trains, validates, tests, and saves a GFNet image classification model.
The model classifies brain images according to whether or not the subject has Alzheimer's disease

Benjamin Thatcher 
s4784738    
"""


import datetime
from typing import Optional
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, accuracy, ModelEma
from functools import partial
import torch.nn as nn

from dataset import get_data_loader
from modules import GFNet
import utils

#import wandb_config


def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast('cuda'):
            outputs = model(samples)
            #loss = criterion(samples, outputs, targets)
            loss = criterion(outputs, targets)


        loss_value = loss.item()
        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
    
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

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

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast('cuda'):
            output = model(images)
            loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_GFNet(dataloaders, num_epochs):
    # Model architecture and hyperparameters
    nb_classes = 2  # Number of output classes (AD/Normal)
    mixup = 0.0
    smoothing = 0.1 

    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #wandb_config.setup_wandb()

    cudnn.benchmark = True

    # Load training and validation datasets
    data_loader_train = dataloaders['train']
    data_loader_val = dataloaders['val']

    # Create model with hard-coded parameters
    print(f"Creating GFNet model with img_size: 224x224")
    model = GFNet(
        img_size=224, patch_size=16, embed_dim=512, depth=19,
        mlp_ratio=4, drop_path_rate=0.25,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ).to(device)

    if device == 'cuda':
        print(torch.cuda.get_device_name(0))
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.1f} GB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024 ** 3:.1f} GB")

    #model_ema = ModelEma(model, decay=args.model_ema_decay) if args.model_ema else None
    model_ema = None
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_parameters}")

    # Optimizer and scheduler
    #optimizer = create_optimizer(args, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    loss_scaler = NativeScaler()

    #lr_scheduler, _ = create_scheduler(args, optimizer)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


    # Loss criterion
    if mixup > 0:
        criterion = SoftTargetCrossEntropy()
    elif smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    print(f"Start training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=0, model_ema=model_ema
        )

        #lr_scheduler.step(epoch)
        lr_scheduler.step()

        # Save checkpoint
        #if args.output_dir:
        #    checkpoint_path = output_dir / f'checkpoint_epoch{epoch}.pth'
        #    utils.save_on_master({
        #        'model': model.state_dict(),
        #        'optimizer': optimizer.state_dict(),
        #        'lr_scheduler': lr_scheduler.state_dict(),
        #        'epoch': epoch,
        #        'scaler': loss_scaler.state_dict(),
        #        'args': args,
        #    }, checkpoint_path)

        # Validate model
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy on validation set: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    print(f'Training time: {str(datetime.timedelta(seconds=int(total_time)))}')


if __name__ == "__main__":
    # Paths to the training and validation datasets
    dataloaders = {
        'train': get_data_loader("/home/groups/comp3710/ADNI/AD_NC/train", batch_size = 32, shuffle = True),
        'val': get_data_loader("/home/groups/comp3710/ADNI/AD_NC/test", batch_size = 32, shuffle = False)
    }

    num_epochs=10
    train_GFNet(dataloaders, num_epochs=num_epochs)
