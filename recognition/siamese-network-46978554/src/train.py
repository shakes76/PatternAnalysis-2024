"""Code for training, validating, testing, and saving model"""

import argparse
import os
from pathlib import Path

import torch
from dataset import MelanomaSiameseReferenceDataset, MelanomaSkinCancerDataset
from modules import SiameseNetwork
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUROC
from torchvision import transforms
from util import contrastive_loss, contrastive_loss_threshold


def train(net, dataset, device, nepochs=10, batch_size=128, start_epoch=0):
    data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4)

    # Put network in training mode
    net = net.to(device)
    net.train()

    # Define loss function and optimiser
    loss_func = contrastive_loss(margin=args.margin)
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

        print("Epoch %3d: %10f" % (epoch + 1, epoch_loss / nbatches))

        # Save model weights and losses every 10 epochs and at the last epoch
        if epoch % 10 == 9 or epoch == nepochs + start_epoch - 1:
            torch.save(torch.Tensor(losses), out_dir / "loss.pt")
            save_obj = {"epoch": epoch, "state_dict": net.state_dict()}
            torch.save(save_obj, out_dir / "checkpoint.pt")


def test(net, dataset, ref_set, device):
    batch_size = 128
    data_loader = DataLoader(dataset, batch_size, shuffle=False)

    # Put network in eval mode
    net = net.to(device)
    net.eval()

    # Define metric
    metric = BinaryAUROC()

    with torch.no_grad():  # Disable gradient computation for efficiency
        for i, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # For AUC, we want the model to output probabilities
            pred = classify(net, x_batch, device, ref_set, prob=True)
            target = y_batch
            metric.update(pred, target)

    return metric.compute()


def classify(net, x, device, ref_set, prob=True):
    batch_size = x.shape[0]
    threshold = contrastive_loss_threshold(margin=0.2)
    preds = []

    with torch.no_grad():
        # Compare x against each reference image and get prediction from net
        # x is batched, so the reference images can't be, i.e. we set batch_size to 1
        ref_set_loader = DataLoader(ref_set, batch_size=1)
        for i, (x_ref, y_ref) in enumerate(ref_set_loader):
            # Replicate x_ref and y_ref to do a batch prediction
            x_ref = x_ref.repeat(batch_size, 1, 1, 1).to(device)
            y_ref = y_ref.repeat(batch_size).to(device)

            # The model returns 0 if the pair is similar, and 1 otherwise
            # However, we need the actual label (0 for benign, 1 for malign)
            # An XOR with the ground truth label (y_ref) will give us the actual label!
            y_hat = threshold(*net(x, x_ref))
            pred = torch.logical_xor(y_hat, y_ref).float()
            preds.append(pred)
        # Stack predictions so that the dimensions are [batch_size, num_ref_imgs]
        preds = torch.stack(preds, dim=1)

    # Probability output is the mean label prediction, which is valid since this is
    # binary classification
    if prob:
        return preds.mean(dim=1)
    # For "hard" classification, use 0.5 as decision boundary
    return (preds.mean(dim=1) >= 0.5).float()


def main():
    torch.manual_seed(3710)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch version", torch.__version__, "on device", device)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(256, padding=4, padding_mode="reflect"),
            transforms.RandomRotation(30),
        ]
    )
    train_set = MelanomaSkinCancerDataset(train=True, transform=train_transform)
    test_set = MelanomaSkinCancerDataset(train=False)
    ref_set = MelanomaSiameseReferenceDataset()

    net = SiameseNetwork(pretrained=args.pretrained)

    if args.action == "train":
        start_epoch = 0
        if args.checkpoint:
            checkpoint = torch.load(
                out_dir / "checkpoint.pt", weights_only=False, map_location=device
            )
            net.load_state_dict(checkpoint["state_dict"])
            start_epoch = checkpoint["epoch"]

        print(f"Training on {device} for {args.epoch} epochs...")
        train(net, train_set, device, nepochs=args.epoch, start_epoch=start_epoch)

        print("Test accuracy:", test(net, test_set, ref_set, device))

    else:  # args.action == "test"
        checkpoint = torch.load(
            out_dir / "checkpoint.pt", weights_only=False, map_location=device
        )
        net.load_state_dict(checkpoint["state_dict"])

        print("Test accuracy:", test(net, test_set, ref_set, device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "test"], help="Training or testing")
    parser.add_argument(
        "-o", "--out", type="str", default="out", help="Training output directory"
    )

    # Training args
    parser.add_argument("-e", "--epoch", type=int, default=10, help="Training epochs")
    parser.add_argument(
        "-p", "--pretrained", action="store_true", help="Train using pretrained ResNet"
    )
    parser.add_argument(
        "-c", "--checkpoint", action="store_true", help="Train from checkpoint"
    )

    # Hyperparameters
    parser.add_argument(
        "-m", "--margin", type=float, default=0.2, help="Margin for contrastive loss"
    )

    args = parser.parse_args()

    out_dir = Path(__file__).parent.parent / args.out
    os.makedirs(out_dir, exist_ok=True)

    main()
