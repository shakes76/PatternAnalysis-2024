"""
Code for training, validating, testing, and saving model

usage: train.py [-h] [-o OUT] [-p] [-m MARGIN] [-e EPOCH] [-l LR] [-b BATCH]
                [--checkpoint-tr CHECKPOINT_TR] [--checkpoint-ts CHECKPOINT_TS]
                {train,test}

positional arguments:
  {train,test}          Training (TR) or testing (TS)

options:
  -h, --help            show this help message and exit
  -o OUT, --out OUT     (TR/TS) output directory, default out
  -p, --pretrained      (TR/TS) whether ResNet base is pretrained, default false
  -m MARGIN, --margin MARGIN
                        (TR/TS) margin for contrastive loss, default 0.2
  -e EPOCH, --epoch EPOCH
                        (TR) epochs, default 10
  -l LR, --lr LR        (TR) learning rate, default 0.001
  -b BATCH, --batch BATCH
                        (TR) batch size, default 128
  --checkpoint-tr CHECKPOINT_TR
                        (TR) checkpoint file (in out directory) to train from, default
                        none (train from scratch)
  --checkpoint-ts CHECKPOINT_TS
                        (TS) checkpoint file (in out directory) to test from, default
                        checkpoint.pt
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from dataset import MelanomaSkinCancerDataset
from modules import SiameseNetwork, init_classifier
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.samplers import MPerClassSampler
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms


def train(
    net,
    dataset,
    device,
    test_func=None,
    nepochs=10,
    batch_size=128,
    start_epoch=0,
):
    # Define dataloader to sample 50/50 split of benign & malignant images in each batch
    labels = dataset.metadata["target"]
    sampler = MPerClassSampler(labels=labels, m=batch_size // 2, batch_size=batch_size)
    data_loader = DataLoader(dataset, batch_size, num_workers=4, sampler=sampler)

    # Put network in training mode
    net = net.to(device)
    net.train()

    # Define loss function and optimiser
    distance = LpDistance(normalize_embeddings=False, p=2, power=1)
    loss_func = ContrastiveLoss(neg_margin=args.margin, distance=distance)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    losses = []
    end_epoch = nepochs + start_epoch
    for epoch in range(start_epoch, end_epoch):
        epoch_loss = 0
        nbatches = 0

        for i, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            embeddings = net(x_batch)
            loss = loss_func(embeddings, y_batch)
            loss.backward()
            optimizer.step()

            losses.append(float(loss))
            epoch_loss += float(loss)
            nbatches += 1

        print("Epoch %3d: %10f" % (epoch + 1, epoch_loss / nbatches))

        # Save model weights and losses every 10 epochs and at the last epoch
        if epoch % 10 == 9 or epoch == end_epoch - 1:
            loss_filename = f"loss-epochs-{start_epoch+1}-{epoch+1}.pt"
            checkpoint_filename = f"checkpoint-epoch-{epoch+1}.pt"
            torch.save(torch.Tensor(losses), out_dir / loss_filename)
            torch.save(net.state_dict(), out_dir / checkpoint_filename)

            # Test the network, if a callback function is provided
            if test_func:
                test_func(net)


def test(net, test_dataset, ref_dataset, device, batch_size=128):
    # Reference images are used for classifying an unseen image - we compare it against
    # every reference image and classify it based on a majority vote
    data_loader = DataLoader(test_dataset, batch_size)

    # Put network in eval mode
    net = net.to(device)
    net.eval()

    # Ground truth (targets) and predictions for test dataset
    preds = []
    preds_proba = []
    targets = []

    clf = init_classifier(net, ref_dataset, device, args.margin)

    with torch.no_grad():  # Disable gradient computation for efficiency
        # Do prediction on test dataset
        for i, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(device)

            embeddings = net(x_batch).cpu()

            # For AUROC, we want the probability of the class with the greater label
            pred = clf.predict(embeddings)
            pred_proba = clf.predict_proba(embeddings)

            preds.append(pred)
            preds_proba.append(pred_proba)
            targets.append(y_batch.numpy())

        preds = np.concatenate(preds)
        preds_proba = np.concatenate(preds_proba)
        targets = np.concatenate(targets)

    report = classification_report(targets, preds, target_names=["benign", "malignant"])
    auc = roc_auc_score(targets, preds_proba)

    return report, auc


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
    train_set = MelanomaSkinCancerDataset(mode="train", transform=train_transform)
    test_set = MelanomaSkinCancerDataset(mode="test")
    ref_set = MelanomaSkinCancerDataset(mode="ref")

    net = SiameseNetwork(pretrained=args.pretrained)

    def test_net(network):
        report, auc = test(network, test_set, ref_set, device)
        print(report)
        print("Test AUROC:", auc)

    if args.action == "train":
        start_epoch = 0
        if args.checkpoint_tr:
            checkpoint_filename = out_dir / args.checkpoint_tr
            checkpoint = torch.load(
                checkpoint_filename, map_location=device, weights_only=True
            )
            net.load_state_dict(checkpoint)

            start_epoch = int(str(checkpoint_filename).split(".")[0].split("-")[-1])

        if start_epoch != 0:
            print(f"Loaded checkpoint (trained for {start_epoch} epochs)")
        print(f"Training on {device} for {args.epoch} epochs...")

        train(
            net,
            train_set,
            device,
            batch_size=args.batch,
            test_func=test_net,
            nepochs=args.epoch,
            start_epoch=start_epoch,
        )

    else:  # args.action == "test"
        checkpoint_filename = out_dir / args.checkpoint_ts
        checkpoint = torch.load(
            checkpoint_filename, map_location=device, weights_only=True
        )
        net.load_state_dict(checkpoint)

        test_net(net)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "test"], help="Training (TR) or testing (TS)")
    parser.add_argument("-o", "--out", type=str, default="out", help="(TR/TS) output directory, default out")

    # Model args, hyperparameters
    parser.add_argument(
        "-p", "--pretrained", action="store_true", help="(TR/TS) whether ResNet base is pretrained, default false"
    )
    parser.add_argument(
        "-m", "--margin", type=float, default=0.2, help="(TR/TS) margin for contrastive loss, default 0.2"
    )

    # Training args
    parser.add_argument("-e", "--epoch", type=int, default=10, help="(TR) epochs, default 10")
    parser.add_argument("-l", "--lr", type=float, default=0.001, help="(TR) learning rate, default 0.001")
    parser.add_argument("-b", "--batch", type=int, default=128, help="(TR) batch size, default 128")
    parser.add_argument(
        "--checkpoint-tr",
        type=str,
        default="",
        help="(TR) checkpoint file (in out directory) to train from, default none (train from scratch)",
    )
    parser.add_argument(
        "--checkpoint-ts",
        type=str,
        default="checkpoint.pt",
        help="(TS) checkpoint file (in out directory) to test from, default checkpoint.pt",
    )

    # fmt: on

    args = parser.parse_args()
    print(args)

    out_dir = Path(__file__).parent.parent / args.out
    os.makedirs(out_dir, exist_ok=True)

    main()
