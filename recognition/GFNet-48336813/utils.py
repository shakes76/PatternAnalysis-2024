"""
This script contains utility functions for configuring and managing the training and evaluation
of the GFNet model. It provides a function for parsing command-line arguments, allowing users to 
easily set parameters such as batch size, learning rate, and dataset paths.

Additionally, it includes a function for plotting and saving training results, such as accuracy 
and loss over epochs, which helps in visualizing model performance during training.

@brief: Utility functions for argument parsing and result plotting for the GFNet model.
@author: Sean Bourchier
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_args_parser():
    """
    Create and return an argument parser for command line arguments.

    Returns:
        argparse.ArgumentParser: Argument parser with training and evaluation options.
    """
    parser = argparse.ArgumentParser('GFNet training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--epochs', default=81, type=int, help='Number of training epochs')

    # Model parameters
    parser.add_argument('--arch', default='gfnet-s', type=str, help='Name of model to train',
                        choices=['gfnet-ti', 'gfnet-xs', 'gfnet-s', 'gfnet-b'])
    parser.add_argument('--input-size', default=224, type=int, help='Input image size')

    # Drop out parameters
    parser.add_argument('--drop', type=float, default=0.1, metavar='PCT',
                        help='Dropout rate (default: 0.0)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use optimizer default)')
    parser.add_argument('--clip-grad', type=float, default=1, metavar='NORM',
                        help='Gradient clipping norm (default: 1.0)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='Learning rate scheduler (default: "cosine")')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='Learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='Learning rate noise on/off epoch percentages')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='Warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='Minimum learning rate (default: 1e-5)')

    # Scheduler timing parameters
    parser.add_argument('--decay-epochs', type=float, default=20, metavar='N',
                        help='Epoch interval to decay LR (default: 30)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='Number of epochs to warmup LR (default: 5)')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='Cooldown epochs after LR schedule ends (default: 10)')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='Patience epochs for Plateau LR scheduler (default: 10)')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='Learning rate decay rate (default: 0.1)')

    # Finetuning parameters
    parser.add_argument('--finetune', default='', help='Path to checkpoint for finetuning')

    # Dataset parameters
    parser.add_argument('--data-path', default='data/', type=str, help='Path to dataset')
    parser.add_argument('--data-set', default='ADNI', type=str, help='Dataset name')

    # Output and device settings
    parser.add_argument('--output_dir', default='outputs/', help='Path to save output')
    parser.add_argument('--device', default='mps', help='Device for training/testing (default: mps)')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--resume', default='', help='Path to checkpoint for resuming training')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='Starting epoch')
    parser.add_argument('--eval', action='store_true', help='Evaluate only without training')

    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfer.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='Disable pinning memory')
    parser.set_defaults(pin_mem=True)

    return parser

def save_plots(architecture, epochs, test_acc, test_loss, current_datetime, plot_dir='plots'):
    """
    Save accuracy and loss plots for training progress.

    Args:
        architecture (str): Model architecture name.
        epochs (list): List of epoch numbers.
        test_acc (list): List of test accuracy values.
        test_loss (list): List of test loss values.
        current_datetime (str): Timestamp for saving plots.
        plot_dir (str): Directory to save plots (default: 'plots').
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plot Accuracy vs Epoch
    plt.figure()
    plt.plot(epochs, test_acc, marker='.')
    plt.title(f'  Accuracy vs Epoch  ({architecture})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.xticks(np.arange(min(epochs), max(epochs) + 1, step=max(1, (max(epochs) + 1) // 10)))
    plt.savefig(os.path.join(plot_dir, f'{current_datetime}_{architecture}_accuracy_vs_epoch.png'))
    plt.close()

    # Plot Test Loss vs Epoch
    plt.figure()
    plt.plot(epochs, test_loss, marker='.')
    plt.title(f'  Loss vs Epoch  ({architecture})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, max(1, (max(test_loss))))
    plt.xticks(np.arange(min(epochs), max(epochs) + 1, step=max(1, (max(epochs) + 1) // 10)))
    plt.savefig(os.path.join(plot_dir, f'{current_datetime}_{architecture}_loss_vs_epoch.png'))
    plt.close()
