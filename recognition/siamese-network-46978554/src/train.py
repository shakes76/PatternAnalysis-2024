"""Code for training, validating, testing, and saving model"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

OUT_DIR = Path(__file__).parent.parent / "out"


def contrastive_loss(margin):
    """
    REF: https://www.sciencedirect.com/topics/computer-science/contrastive-loss
    """

    def f(x1, x2, y):
        dist = F.pairwise_distance(x1, x2)
        dist_sq = torch.pow(dist, 2)

        loss = (1 - y) * dist_sq + y * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
        loss = torch.mean(loss / 2.0, dim=0)

        return loss

    return f


def train(
    net, dataset, device, nepochs=10, batch_size=128, num_workers=0, start_epoch=0
):
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    net = net.to(device)
    net.train()

    loss_func = contrastive_loss(margin=1.0)
    optimizer = torch.optim.Adam(net.parameters())

    losses = []
    for epoch in range(start_epoch, nepochs + start_epoch):
        epoch_loss = 0
        nbatches = 0

        for i, (x1_batch, x2_batch, y_batch) in enumerate(data_loader):
            x1_batch = x1_batch.to(device)
            x2_batch = x2_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            out1, out2 = net(x1_batch, x2_batch)
            loss = loss_func(out1, out2, y_batch)
            loss.backward()
            optimizer.step()

            losses.append(float(loss))
            epoch_loss += float(loss)
            nbatches += 1

        if epoch % 10 == 9:
            torch.save(torch.Tensor(losses), OUT_DIR / "loss.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                },
                OUT_DIR / "checkpoint.pt",
            )

        print("Epoch %3d: %10f" % (epoch + 1, epoch_loss / nbatches))
