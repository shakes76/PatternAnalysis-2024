"""
This script is used to train the GFNet model for binary classification, specifically distinguishing between
AD (Alzheimer's Disease) and NC (Normal Cognition). It supports training on various devices (CPU, CUDA, MPS),
loads training and validation datasets, configures the model architecture, optimizer, and learning rate scheduler,
and saves checkpoints during the training process. The script also evaluates the model after each epoch 
and tracks the training progress using accuracy and loss metrics.

@brief: Training script for GFNet model with binary classification (AD vs NC).
@author: Sean Bourchier
"""

import json
import time
import datetime
from datetime import datetime as now
import argparse
from functools import partial
import torch
import torch.nn as nn
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from pathlib import Path

from utils import get_args_parser, save_plots
from dataset import build_dataset
from modules import GFNet

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler, max_norm=0, set_training_mode=True):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        criterion (torch.nn.CrossEntropyLoss): Loss function.
        data_loader (Iterable): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        device (torch.device): Device to run training on (CPU, GPU).
        epoch (int): Current training epoch.
        loss_scaler: Loss scaler for mixed precision training.
        max_norm (float): Gradient clipping max norm.
        set_training_mode (bool): Whether to set model to training mode.

    Returns:
        dict: Average loss and learning rate for the epoch.
    """
    model.train(set_training_mode)
    total_loss = 0.0
    total_samples = 0
    print_freq = 10  # Frequency for printing training progress

    for i, (samples, targets) in enumerate(data_loader):
        samples, targets = samples.to(device), targets.to(device)  # Move data to the appropriate device
        outputs = model(samples)  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss

        loss_value = loss.item()
        total_loss += loss_value * samples.size(0)
        total_samples += samples.size(0)

        optimizer.zero_grad()  # Zero gradients before backward pass
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters())  # Perform backward pass and optimizer step
        if device.type == 'cuda':
            torch.cuda.synchronize()

        if i % print_freq == 0:
            print(f'Epoch [{epoch}], Step [{i}/{len(data_loader)}], Loss: {loss_value:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    # Calculate average loss
    avg_loss = total_loss / total_samples
    return {'loss': avg_loss, 'lr': optimizer.param_groups[0]["lr"]}


@torch.no_grad()
def evaluate(data_loader, model, device):
    """
    Evaluate the model on the validation dataset.

    Args:
        data_loader (Iterable): Validation data loader.
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): Device to run evaluation on (CPU, GPU).

    Returns:
        dict: Average loss and accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    model.eval()  # Set the model to evaluation mode

    for images, targets in data_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss
        total_loss += loss.item() * images.size(0)

        # Calculate accuracy
        _, predicted = outputs.max(1)
        total_correct += (predicted == targets).sum().item()
        total_samples += targets.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100

    print(f'Test - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return {'loss': avg_loss, 'accuracy': accuracy}


def setup_device():
    """
    Setup the device for training (CPU, GPU, MPS).

    Returns:
        torch.device: Configured device.
    """
    # Check for available hardware and return appropriate device
    return torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

def setup_dataloaders(args):
    """
    Prepare the data loaders for training and validation.

    Args:
        args: Parsed command line arguments.

    Returns:
        tuple: Training and validation data loaders, number of classes.
    """
    # Build datasets for training and validation
    dataset_train, args.nb_classes = build_dataset('train', args=args)
    dataset_val, _ = build_dataset('val', args=args)
    
    # Define samplers for training and validation datasets
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    # Create data loaders
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
    return data_loader_train, data_loader_val

def save_checkpoint(epoch, model, optimizer, lr_scheduler, loss_scaler, args, output_dir, best=False):
    """
    Save a checkpoint of the model.

    Args:
        epoch (int): Current epoch.
        model (torch.nn.Module): The model.
        optimizer (torch.optim.Optimizer): The optimizer.
        lr_scheduler: Learning rate scheduler.
        loss_scaler: Loss scaler for mixed precision training.
        args: Command line arguments.
        output_dir (Path): Directory to save checkpoints.
        best (bool): Flag to save as the best checkpoint.
    """
    # Define the checkpoint path based on whether it's the best checkpoint or last checkpoint
    checkpoint_path = output_dir / ('checkpoint_best.pth' if best else 'checkpoint_last.pth')
    
    # Save the state of the model, optimizer, and other components
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'args': args,
    }, checkpoint_path)

def main():
    """
    Main function to run training and evaluation.
    """
    current_time = now.now().strftime('%Y-%m-%d %H:%M:%S')  # Get current timestamp for plots
    parser = argparse.ArgumentParser('GFNet training and validation script', parents=[get_args_parser()])
    args = parser.parse_args()  # Parse command-line arguments
    
    device = setup_device()  # Setup the device (CPU, GPU, or MPS)
    data_loader_train, data_loader_val = setup_dataloaders(args)
    
    # Create model instance
    print(f"Creating model: {args.arch}")
    model = create_model(args)
    model.to(device)
    print("Model ready.")
    
    # Setup optimizer and scheduler
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Track results for plotting
    test_acc_list, test_loss_list, epoch_list = [], [], []
    output_dir = Path(args.output_dir)
    start_time = time.time()
    max_accuracy = 0.0
    
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        start_time_epoch = time.time()  # Start timer for the epoch
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, set_training_mode=args.finetune == ''
        )

        lr_scheduler.step(epoch)  # Update the learning rate
        save_checkpoint(epoch, model, optimizer, lr_scheduler, loss_scaler, args, output_dir)

        # Evaluate model on validation data
        test_stats = evaluate(data_loader_val, model, device)
        test_acc_list.append(test_stats['accuracy'])
        test_loss_list.append(test_stats['loss'])
        epoch_list.append(epoch)
        max_accuracy = max(max_accuracy, test_stats["accuracy"])

        print(f"Accuracy of the network on the {len(data_loader_val.dataset)} test images: {test_stats['accuracy']:.1f}%")
        print(f'Max accuracy: {max_accuracy:.2f}%')

        # Save the best model
        if max_accuracy == test_stats["accuracy"]:
            save_checkpoint(epoch, model, optimizer, lr_scheduler, loss_scaler, args, output_dir, best=True)
        
        # Log and save results
        log_training_stats(epoch, train_stats, test_stats, args, output_dir, current_time)
        save_plots(args.arch, epoch_list, test_acc_list, test_loss_list, current_time)
        print('Single epoch time: {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time_epoch)))))
    
    print('Total training time: {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))

def create_model(args):
    """
    Create and return a model instance based on architecture choice.

    Args:
        args: Parsed command line arguments.

    Returns:
        torch.nn.Module: Configured model.
    """
    # Create the model based on the specified architecture
    if args.arch == 'gfnet-xs':
        return GFNet(img_size=args.input_size, patch_size=16, embed_dim=384, depth=12, mlp_ratio=4,
                     norm_layer=partial(nn.LayerNorm, eps=1e-6))
    elif args.arch == 'gfnet-ti':
        return GFNet(img_size=args.input_size, patch_size=16, embed_dim=256, depth=12, mlp_ratio=4,
                     norm_layer=partial(nn.LayerNorm, eps=1e-6))
    elif args.arch == 'gfnet-s':
        return GFNet(img_size=args.input_size, patch_size=16, embed_dim=384, depth=19, mlp_ratio=4,
                     drop_path_rate=0.15, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    elif args.arch == 'gfnet-b':
        return GFNet(img_size=args.input_size, patch_size=16, embed_dim=512, depth=19, mlp_ratio=4,
                     drop_path_rate=0.25, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    else:
        raise NotImplementedError(f"Model architecture {args.arch} not supported.")

def log_training_stats(epoch, train_stats, test_stats, args, output_dir, time):
    """
    Log training statistics to a file.

    Args:
        epoch (int): Current epoch.
        train_stats (dict): Training statistics.
        test_stats (dict): Evaluation statistics.
        args: Parsed command line arguments.
        output_dir (Path): Directory to save logs.
    """
    # Compile all statistics into a log
    log_stats = {
        'time': time,
        'arch': args.arch,
        'epoch': epoch,
        **{f'train_{k}': v for k, v in train_stats.items()},
        **{f'test_{k}': v for k, v in test_stats.items()},
    }
    if args.output_dir:
        # Write log to file
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

if __name__ == "__main__":
    main()
