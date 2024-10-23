import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json

from dataset import adni_data_load
from gfnet import GFNet, _cfg
from functools import partial


from visualise import visualize_batch

def main():
    # adni_dir = "/home/reuben/Documents/GFNet_testing/ADNI_AD_NC_2D/AD_NC"
    adni_dir = "/home/reuben/Documents/datasets/ADNI_AD_NC_2D/AD_NC"
    dataset_val, dataloader_val = adni_data_load(adni_dir, verbose=True, test_set=True) 

    # visualize_batch(dataloader_val, ["NC", "AD"])

    model = GFNet(
        img_size=224, 
        patch_size=16, embed_dim=384, depth=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=2
    )

    # model_path = "/home/reuben/Documents/remote-repos/PatternAnalysis-2024/recognition/GFNet-ADNI-46986830/pretrained/adni_gfnet-xs_10epoch_best.pth"
    model_path = "/home/reuben/MEGA/uni/Y4-S2/COMP3710/project/pretrained_models/adni_gfnet-xs_50epoch_best.pth"
    model.default_cfg = _cfg()

    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    print('## model has been successfully loaded')

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    # model = model.cuda()
    model.to(device=device)

    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    validate(dataloader_val, model, criterion, device)

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
        # fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        # return fmtstr.format(**self.__dict__)
        # return fmtstr.format(name=self.name, val=self.val, avg=self.avg)
        output = self.name + " " + str(self.val) + " (" + str(self.avg) + ')'
        return output

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
    # top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')


    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # images = images.cuda()
            images = images.to(device=device)
            # target = target.cuda()
            target = target.to(device=device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1= accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #       .format(top1=top1, top5="shjit"))

        print("Accuracy: ", top1.avg[0])

    return top1.avg

main()