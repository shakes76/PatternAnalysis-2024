import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from functools import partial

from dataset import build_dataset
from modules import GFNet, _cfg

def get_args_parser():
    parser = argparse.ArgumentParser('GFNet testing script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--arch', default='gfnet-xs', type=str, help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--data-path', default='data/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='ADNI', choices=['ADNI', 'CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Dataset name')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model-path', default='outputs/checkpoint_best.pth', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    return parser


def main(args):
    # Check device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cudnn.benchmark = True
        print("CUDA is available. Using GPU with CUDA.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available. Using GPU with Metal (M1/M2).")
    else:
        device = torch.device("cpu")
        print("Neither CUDA nor MPS are available. Using CPU.")
        
    # Build test dataset
    dataset_test, _ = build_dataset(split='test', args=args)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=128,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # Assign model instance
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
            patch_size=16, embed_dim=384, depth=19, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    elif args.arch == 'gfnet-b':
        model = GFNet(
            img_size=args.input_size, 
            patch_size=16, embed_dim=512, depth=19, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    else:
        raise NotImplementedError

    # Configure model and load trained model weights
    model_path = args.model_path
    model.default_cfg = _cfg()
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    print('## model has been successfully loaded')

    model = model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)

    # Assign testing criteria
    if torch.cuda.is_available():
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Test model on test data
    validate(data_loader_test, model, criterion, device)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')


    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            
            # Get the number of classes from the output
            num_classes = output.size(1)

            # Ensure topk doesn't exceed the number of classes
            maxk = min(5, num_classes)  # Use top 5, or fewer if there are fewer classes

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, maxk))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0:
                progress.display(i)

        # TODO: this should also be done with the cuda
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GFNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
