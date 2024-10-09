"""Code for training, validating, testing, and saving model"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from dataset import MelanomaSiameseReferenceDataset, MelanomaSkinCancerDataset
from modules import SiameseNetwork
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from util import (contrastive_loss, contrastive_loss_threshold,
                  plot_pca_embeddings)


def train(
    net,
    dataset,
    device,
    test_func=None,
    nepochs=10,
    batch_size=128,
    start_epoch=0,
):
    data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4)

    # Put network in training mode
    net = net.to(device)
    net.train()

    # Define loss function and optimiser
    loss_func = contrastive_loss(margin=args.margin)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

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

        print("Epoch %3d: %10f" % (start_epoch + epoch + 1, epoch_loss / nbatches))

        # Save model weights and losses every 10 epochs and at the last epoch
        if epoch % 10 == 9 or epoch == nepochs + start_epoch - 1:
            torch.save(torch.Tensor(losses), out_dir / "loss.pt")
            save_obj = {"epoch": epoch, "state_dict": net.state_dict()}
            torch.save(save_obj, out_dir / "checkpoint.pt")

            if test_func:
                test_func(net)


def test(net, dataset, ref_set, device):
    batch_size = 128
    data_loader = DataLoader(dataset, batch_size, shuffle=False)

    # Reference set is used for classification - we compare each new image to an
    # image with an already known label

    # Put network in eval mode
    net = net.to(device)
    net.eval()

    targets_ref = []
    outs_ref = []

    preds = []
    preds_proba = []
    targets = []
    outs = []

    with torch.no_grad():  # Disable gradient computation for efficiency
        # Run reference images through network to get reference outputs
        ref_set_loader = DataLoader(ref_set, batch_size=8)
        for i, (x_ref_batch, y_ref_batch) in enumerate(ref_set_loader):
            x_ref_batch = x_ref_batch.to(device)
            y_ref_batch = y_ref_batch.to(device)
            out_ref = net.forward_one(x_ref_batch)

            targets_ref.append(y_ref_batch)
            outs_ref.append(out_ref)

        targets_ref = torch.concat(targets_ref)
        outs_ref = torch.concat(outs_ref)

        # Do prediction on test set
        for i, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(device)
            out = net.forward_one(x_batch)
            # For AUC, we want the model to output probabilities
            pred_proba = classify(out, outs_ref, targets_ref, device, prob=True)
            # For classification report, we want the hard classifications
            pred = (pred_proba >= 0.5).float()

            targets.append(y_batch.numpy())
            outs.append(out.cpu().numpy())
            preds.append(pred.cpu().numpy())
            preds_proba.append(pred_proba.cpu().numpy())

        targets = np.concatenate(targets)
        outs = np.concatenate(outs)
        preds = np.concatenate(preds)
        preds_proba = np.concatenate(preds_proba)

    plot_pca_embeddings(outs, targets)

    report = classification_report(targets, preds, target_names=["benign", "malignant"])
    auc = roc_auc_score(targets, preds_proba)

    return report, auc


def classify(out, outs_ref, targets_ref, device, prob=True):
    batch_size = out.shape[0]
    threshold = contrastive_loss_threshold(margin=args.margin)

    preds = []
    for out_ref, target_ref in zip(outs_ref, targets_ref):
        out_ref = out_ref.repeat(batch_size, 1)
        target_ref = target_ref.repeat(batch_size)
        pair_pred = threshold(out, out_ref)

        # The model returns 0 if the pair is similar, and 1 otherwise
        # However, we need the actual label (0 for benign, 1 for malign)
        # An XOR with the ground truth label (y_ref) will give us the actual label!
        pred = torch.logical_xor(pair_pred, target_ref).float()
        preds.append(pred)

    # Stack predictions so that the dimensions are [batch_size, num_ref_imgs]
    preds = torch.stack(preds, dim=1)

    # Probability output is the mean label prediction
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
    ref_set = MelanomaSiameseReferenceDataset(size=16)

    net = SiameseNetwork(pretrained=args.pretrained)

    def test_net(network):
        report, auc = test(network, test_set, ref_set, device)
        print(report)
        print("Test AUROC:", auc)

    if args.action == "train":
        start_epoch = 0
        if args.checkpoint_tr:
            checkpoint = torch.load(
                out_dir / args.checkpoint_tr, weights_only=False, map_location=device
            )
            net.load_state_dict(checkpoint["state_dict"])
            start_epoch = checkpoint["epoch"]

        if start_epoch == 0:
            print(f"Training on {device} for {args.epoch} epochs...")
        else:
            print(
                f"Training on {device} from checkpoint ({start_epoch} epochs) for {args.epoch} epochs..."
            )
        train(
            net,
            train_set,
            device,
            batch_size=args.batch,
            checkpoint_func=test_net,
            nepochs=args.epoch,
            start_epoch=start_epoch,
        )

    else:  # args.action == "test"
        checkpoint = torch.load(
            out_dir / args.checkpoint_ts, weights_only=False, map_location=device
        )
        net.load_state_dict(checkpoint["state_dict"])

        test_net(net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "test"], help="Training or testing")
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="out",
        help="(TR/TS) output directory, default out",
    )

    # Model args, hyperparameters
    parser.add_argument(
        "-p",
        "--pretrained",
        action="store_true",
        help="(TR/TS) whether ResNet base is pretrained or not",
    )
    parser.add_argument(
        "-m",
        "--margin",
        type=float,
        default=0.2,
        help="(TR/TS) margin for contrastive loss, default 0.2",
    )

    # Training args
    parser.add_argument(
        "-e", "--epoch", type=int, default=10, help="(TR) epochs, default 10"
    )
    parser.add_argument(
        "-l",
        "--lr",
        type=float,
        default=0.001,
        help="(TR) learning rate, default 0.001",
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=128, help="(TR) batch size, default 128"
    )
    parser.add_argument(
        "--checkpoint-tr",
        type=str,
        default="",
        help="(TR) checkpoint file to train from, default none (train from scratch)",
    )
    parser.add_argument(
        "--checkpoint-ts",
        type=str,
        default="checkpoint.pt",
        help="(TS) checkpoint file (in output directory) to test from, default checkpoint.pt",
    )

    args = parser.parse_args()
    print(args)

    out_dir = Path(__file__).parent.parent / args.out
    os.makedirs(out_dir, exist_ok=True)

    main()
