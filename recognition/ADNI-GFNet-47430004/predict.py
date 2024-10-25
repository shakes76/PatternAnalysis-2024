"""
This file is used to assess the accuracy of a pre-trained model stored in
test/model/GFNet.pth, by letting the model run through test set once.
The script then plots graphs representing the accuracy of the model over
one test set, with the y value representing accuracy per batch.

Got inspiration from infer.py file of the following github repo:
https://github.com/shakes76/GFNet
"""

import torch
from modules import GFNet
from dataset import ADNIDataset, get_dataloaders
import time
import torch
import torch.backends.cudnn as cudnn
import os

import matplotlib.pyplot as plt

out_dir = 'test/model/run/figs/'
model_path = "test/model/GFNet.pth"

class AverageMeter(object):
    # This class is taken from the original source
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
    # This class is taken from the original source
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
    # This function is taken from the original source
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

def validate(test_loader, model, criterion):
    # This function is taken from the original source
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()
    acc1list = []

    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1],
        prefix='Test: ')


    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            acc1list.append(acc1.cpu())

            if i % 100 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return acc1list

if __name__ == '__main__':
    # This function was written by me, getting some inspiration from the original source
    print("Main of Predict")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    cudnn.benchmark = True
    _, test_loader = get_dataloaders(None)

    model = GFNet(num_classes=2, in_chans=1)
    model.to(device)


    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    print('## model has been successfully loaded')

    model = model.cuda()

    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    acc1list = validate(test_loader, model, criterion)
    plt.figure(1)
    plt.title('Result')
    plt.xlabel('Batch Number')
    plt.ylabel('Acc @ 1')
    plt.plot([x for x in range(len(acc1list))], acc1list)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(out_dir + "test")
    plt.show()