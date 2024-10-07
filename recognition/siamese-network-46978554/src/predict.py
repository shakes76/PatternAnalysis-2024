"""Example usage of trained model"""

import torch
from torch.utils.data import DataLoader
from train import contrastive_loss_threshold


def classify(net, x, device, ref_set):
    batch_size = x.shape[0]
    threshold = contrastive_loss_threshold(margin=1.0)
    with torch.no_grad():
        preds = []
        ref_set_loader = DataLoader(ref_set, batch_size=1)
        for i, (x_batch, y_batch) in enumerate(ref_set_loader):
            x_batch = x_batch.repeat(128, 1, 1, 1).to(device)
            y_batch = y_batch.repeat(128, 1, 1, 1).to(device)

            # y_batch is the actual label of the image.
            # net() returns 0 if the pair are similar, and 1 otherwise.
            # To get the label prediction from net(), use an XOR!
            y_hat = threshold(*net(x, x_batch))
            pred = torch.logical_xor(y_hat, y_batch).float()
            preds.append(pred)
        preds = torch.concat(preds)

    return preds.mean() >= 0.5
