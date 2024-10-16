import os
import argparse
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import torch

def get_args_parser():
    """
    Create and return an argument parser for command line arguments.

    Returns:
        argparse.ArgumentParser: Argument parser with training and evaluation options.
    """
    parser = argparse.ArgumentParser('GFNet training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--epochs', default=51, type=int, help='Number of training epochs')

    # Model parameters
    parser.add_argument('--arch', default='gfnet-b', type=str, help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='Input image size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
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
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
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


def save_plots(architecture, epochs, test_acc, test_loss, current_datetime, plot_dir='images'):
    """
    Save accuracy and loss plots for training progress.

    Args:
        architecture (str): Model architecture name.
        epochs (list): List of epoch numbers.
        test_acc (list): List of test accuracy values.
        test_loss (list): List of test loss values.
        current_datetime (str): Timestamp for saving plots.
        plot_dir (str): Directory to save plots (default: 'images').
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plot Accuracy vs Epoch
    plt.figure()
    plt.plot(epochs, test_acc, marker='.')
    plt.title(f'Accuracy vs Epoch ({architecture})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.xticks(np.arange(min(epochs), max(epochs) + 1, step=max(1, (max(epochs) + 1) // 20)))
    plt.savefig(os.path.join(plot_dir, f'{current_datetime}_{architecture}_accuracy_vs_epoch.png'))
    plt.close()

    # Plot Test Loss vs Epoch
    plt.figure()
    plt.plot(epochs, test_loss, marker='.')
    plt.title(f'Loss vs Epoch ({architecture})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.xticks(np.arange(min(epochs), max(epochs) + 1, step=max(1, (max(epochs) + 1) // 20)))
    plt.savefig(os.path.join(plot_dir, f'{current_datetime}_{architecture}_loss_vs_epoch.png'))
    plt.close()


class SmoothedValue:
    """Tracks a series of values and provides smoothed metrics over a window or the global series average."""

    def __init__(self, window_size=20, fmt=None):
        """
        Initialize a SmoothedValue object.

        Args:
            window_size (int): The size of the window for smoothing.
            fmt (str): The format string for displaying the smoothed value.
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """
        Update the current value and maintain a rolling average.

        Args:
            value (float): The new value to add.
            n (int): The number of occurrences of the value (default: 1).
        """
        self.deque.append(value)
        self.total += value * n
        self.count += n

    @property
    def median(self):
        """Return the median of the values."""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """Return the average of the values."""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """Return the global average of all values."""
        return self.total / self.count

    @property
    def max(self):
        """Return the maximum value seen."""
        return max(self.deque)

    @property
    def value(self):
        """Return the most recent value."""
        return self.deque[-1]

    def __str__(self):
        """Return a formatted string representation of the smoothed values."""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


class MetricLogger:
    """Logger for tracking metrics during training and evaluation, with smoothing and formatted output."""

    def __init__(self, delimiter="\t"):
        """
        Initialize a MetricLogger object.

        Args:
            delimiter (str): Delimiter for separating log messages.
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        Update the logger with new values.

        Args:
            **kwargs: Named values to log (e.g., loss=0.5).
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """Get a metric value by its name."""
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        """Return a formatted string of all tracked metrics."""
        return self.delimiter.join(f"{name}: {str(meter)}" for name, meter in self.meters.items())

    def add_meter(self, name, meter):
        """
        Add a custom meter for tracking.

        Args:
            name (str): Name of the metric.
            meter (SmoothedValue): Instance of a SmoothedValue for custom metric tracking.
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        Log metrics at regular intervals during iteration.

        Args:
            iterable (Iterable): The iterable object to loop through.
            print_freq (int): Frequency of logging.
            header (str): Optional header for the log output.

        Yields:
            The elements of the iterable, logging metrics at specified intervals.
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)')
