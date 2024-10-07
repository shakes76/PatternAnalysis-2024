"""Code for training, validating, testing, and saving model"""

import argparse
import os

import torch
from dataset import MelanomaSiameseReferenceDataset, MelanomaSkinCancerDataset
from modules import SiameseNetwork
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUROC
from torchvision import transforms
from util import OUT_DIR, contrastive_loss, contrastive_loss_threshold


def train(net, dataset, device, nepochs=10, batch_size=128, start_epoch=0):
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Put network in training mode
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

        # Save model weights and losses every 10 epochs
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


def test(net, dataset, ref_set, device):
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Put network in eval mode
    net = net.to(device)
    net.eval()
    metric = BinaryAUROC()

    with torch.no_grad():  # Disable gradient computation for efficiency
        for i, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = classify(net, x_batch, device, ref_set)
            target = y_batch
            metric.update(pred, target)

    return metric.compute()


def classify(net, x, device, ref_set, prob=True):
    batch_size = x.shape[0]
    threshold = contrastive_loss_threshold(margin=1.0)
    with torch.no_grad():
        preds = []
        ref_set_loader = DataLoader(ref_set, batch_size=1)
        for i, (x_batch, y_batch) in enumerate(ref_set_loader):
            x_batch = x_batch.repeat(batch_size, 1, 1, 1).to(device)
            y_batch = y_batch.repeat(batch_size).to(device)

            # y_batch is the actual label of the image.
            # net() returns 0 if the pair are similar, and 1 otherwise.
            # To get the label prediction from net(), we do an XOR
            y_hat = threshold(*net(x, x_batch))
            pred = torch.logical_xor(y_hat, y_batch).float()
            preds.append(pred)
        preds = torch.stack(preds, dim=1)

    if prob:
        return preds.mean(dim=1)
    return (preds.mean(dim=1) >= 0.5).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "test"], help="Training or testing")
    parser.add_argument("-e", "--epoch", type=int, default=10, help="Training epochs")
    parser.add_argument(
        "-p", "--pretrained", action="store_true", help="Train using pretrained ResNet"
    )
    parser.add_argument(
        "-c", "--checkpoint", action="store_true", help="Train from checkpoint"
    )
    args = parser.parse_args()

    torch.manual_seed(3710)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch version", torch.__version__, "on device", device)

    os.makedirs(OUT_DIR, exist_ok=True)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(256, padding=4, padding_mode="reflect"),
        ]
    )

    net = SiameseNetwork(pretrained=args.pretrained)
    train_set = MelanomaSkinCancerDataset(train=True, transform=train_transform)
    ref_set = MelanomaSiameseReferenceDataset()
    test_set = MelanomaSkinCancerDataset(train=False)

    if args.action == "train":
        start_epoch = 0
        if args.checkpoint:
            checkpoint = torch.load(
                OUT_DIR / "checkpoint.pt", weights_only=False, map_location=device
            )
            net.load_state_dict(checkpoint["state_dict"])
            start_epoch = checkpoint["epoch"]

        print(f"Training on {device} for {args.epoch} epochs...")
        train(net, train_set, device, nepochs=args.epoch, start_epoch=start_epoch)

        print("Test accuracy:", test(net, test_set, ref_set, device))

    else:  # args.action == "test"
        checkpoint = torch.load(
            OUT_DIR / "checkpoint.pt", weights_only=False, map_location=device
        )
        net.load_state_dict(checkpoint["state_dict"])

        print("Test accuracy:", test(net, test_set, ref_set, device))


if __name__ == "__main__":
    main()
