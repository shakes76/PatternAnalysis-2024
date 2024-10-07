"""Code for training, validating, testing, and saving model"""

import argparse
import os

import torch
import torch.nn.functional as F
from dataset import MelanomaSkinCancerDataset
from modules import SiameseNetwork
from torch.utils.data import DataLoader
from util import OUT_DIR


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "test"], help="Training or testing")
    parser.add_argument(
        "--tr-nepoch", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--tr-resume",
        action="store_true",
        help="Whether to train from existing weights",
    )
    args = parser.parse_args()

    torch.manual_seed(3710)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch version", torch.__version__, "on device", device)

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.action == "train":
        net = SiameseNetwork()
        train_set = MelanomaSkinCancerDataset(train=True)

        start_epoch = 0
        if args.tr_resume:
            checkpoint = torch.load(
                OUT_DIR / "checkpoint.pt", weights_only=False, map_location=device
            )
            net.load_state_dict(checkpoint["state_dict"])
            start_epoch = checkpoint["epoch"]

        print(f"Training on {device} for {args.tr_nepoch} epochs...")
        train(
            net,
            train_set,
            device,
            nepochs=args.tr_nepoch,
            start_epoch=start_epoch,
            num_workers=4,
        )


if __name__ == "__main__":
    main()
