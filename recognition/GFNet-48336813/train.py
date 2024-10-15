
import sys
import math
import json
import time
import datetime
from datetime import datetime as now
import argparse
from functools import partial
import torch
import torch.nn as nn
import torch.distributed as dist
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma, accuracy
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.data import Mixup
from typing import Iterable, Optional
from pathlib import Path

import utils
from utils import get_args_parser, save_on_master, is_main_process, save_plots
from dataset import build_dataset
from modules import GFNet


def train_one_epoch(model: torch.nn.Module, criterion: LabelSmoothingCrossEntropy,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
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

        if device == "cuda":
            with torch.autocast(device_type="cuda"):
                outputs = model(samples)
                loss = criterion(outputs, targets)
        else:
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
    
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
            
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
        if device == "cuda":
            with torch.autocast(device_type="cuda"):
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



if __name__ == "__main__":

    # Get arguments (stored in utils.py)
    parser = argparse.ArgumentParser('GFNet training and validation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    # Set device
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Get datasets
    dataset_train, args.nb_classes = build_dataset('train', args=args)
    dataset_val, _ = build_dataset('val', args=args)
    
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Set data augmentation technique mixup parameters
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print('standard mix up')
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    else:
        print('mix up is not used')
 
    # Create model instance based off parameters
    print(f"Creating model: {args.arch}")
    if args.arch == 'gfnet-xs':
        model = GFNet(
            img_size=args.input_size, 
            patch_size=16, embed_dim=384, depth=12, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    elif args.arch == 'gfnet-ti':
        model = GFNet(
            img_size=args.input_size, 
            patch_size=16, embed_dim=256, depth=12, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    elif args.arch == 'gfnet-s':
        model = GFNet(
            img_size=args.input_size, 
            patch_size=16, embed_dim=384, depth=19, mlp_ratio=4, drop_path_rate=0.15,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    elif args.arch == 'gfnet-b':
        model = GFNet(
            img_size=args.input_size, 
            patch_size=16, embed_dim=512, depth=19, mlp_ratio=4, drop_path_rate=0.25,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    else:
        raise NotImplementedError
    model.to(device)
    print("Model ready.")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # Set model ema parameters if used (for training stabilisation)
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    # Set optimiser and lr scheduler from parameters
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # Set loss based off parameters
    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    # For plotting results
    test_acc_list = []
    test_loss_list = []
    epoch_list = []
        
    # Training loop
    print(f"Start training for {args.epochs} epochs")
    current_datetime = now.now().strftime('%Y-%m-%d %H:%M:%S')
    output_dir = Path(args.output_dir)
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
         
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.finetune == ''  # keep in eval mode during finetuning
        )

        lr_scheduler.step(epoch)

        # Checkpoint saving
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_last.pth']
            for checkpoint_path in checkpoint_paths:
                if model_ema is not None:
                    save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
                else:
                    save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
        
        if (epoch + 1) % 20 == 0:
            file_name = 'checkpoint_epoch%d.pth' % epoch
            checkpoint_path = output_dir / file_name
            if model_ema is not None:
                save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
            else:
                save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        # Model validation after each epoch
        test_stats = evaluate(data_loader_val, model, device)
        # For plotting results
        test_acc_list.append(test_stats['acc1'])
        test_loss_list.append(test_stats['loss'])
        epoch_list.append(epoch)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        # Always store best checkpoint
        if max_accuracy == test_stats["acc1"]:
            checkpoint_path = output_dir / 'checkpoint_best.pth'
            if model_ema is not None:
                save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
            else:
                save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        # Stats logged in log.txt
        log_stats = {   'dtc': current_datetime,
                        'arch': args.arch,
                        'epoch': epoch,
                        **{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'n_parameters': n_parameters}

        if args.output_dir and is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                
        # Save plots
        save_plots(architecture=args.arch, epochs=epoch_list, 
                   test_acc=test_acc_list, test_loss=test_loss_list,
                   current_datetime=current_datetime)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))