# The source code for training, validating, testing and saving your model. The model
# should be imported from “modules.py” and the data loader should be imported from “dataset.py”.
# Make sure to plot the losses and metrics during training

import os

# import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules import SiameseNetwork
from dataset_3 import Dataset
from predict import PredictData

if __name__ == "__main__":
    # Set device to CUDA if a CUDA device is available, else CPU
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parameter declaration
    learning_rate = 0.0000001
    num_epochs = 30
    backbone = "resnet18"  # the feature extraction model we are using
    save_after = 10  # save the model's image every {20} epoch

    # path to write things to - includes summary writer, checkpoints,
    out_path = r'C:\Users\sebas\project\outputs'
    #out_path = "~/project/outputs/"

    """
    # get data from the dataset.py file
    #images_path = "~/.kaggle/train-image/image" -> for rangpur
    #csv_path = "~/.kaggle/train-metadata.csv" -> for rangpur
    images_path = "~/archive/train-image/image
    csv_path = "~/archive/train-metadata.csv
    """

    data_path = r'C:\Users\sebas\archive\large_dataset'
    #data_path = "~/.kaggle/"

    dataset = Dataset(data_path)

    # Get the splits
    train_dataset = dataset.get_split('train')
    val_dataset = dataset.get_split('val')
    test_dataset = dataset.get_split('test')

    # Create DataLoader for each split
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=4)

    model = SiameseNetwork()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = torch.nn.BCELoss()

    # summary writer - keeps track of training progression/visualising training progression
    writer = SummaryWriter(os.path.join(out_path, "summary"))

    best_val = 10000000000  # big value -> any new validation loss will be better

    for epoch in range(num_epochs):
        print("[{} / {}]".format(epoch, num_epochs))
        # puts the model into training mode -> enables dropout, and batch normalization uses mini-batch statistics
        model.train()

        losses = []
        correct = 0
        total = 0

        # Training Loop Start
        for (img1, img2), target, (class1, class2) in train_loader:
            img1, img2, target = map(lambda x: x.to(device), [img1, img2, target])

            similarity = model(img1, img2)
            loss = criterion(similarity, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            correct += torch.count_nonzero(target == (similarity > 0.5)).item()
            total += len(target)

        writer.add_scalar('train_loss', sum(losses) / len(losses), epoch)
        writer.add_scalar('train_acc', correct / total, epoch)

        print("\tTraining: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses) / len(losses), correct / total))
        # Training Loop End

        # Evaluation Loop Start
        model.eval()

        losses = []
        correct = 0
        total = 0

        for (img1, img2), target, (class1, class2) in val_loader:
            img1, img2, target = map(lambda x: x.to(device), [img1, img2, target])

            similarity = model(img1, img2)
            loss = criterion(similarity, target)

            losses.append(loss.item())
            correct += torch.count_nonzero(target == (similarity > 0.5)).item()
            total += len(target)

        val_loss = sum(losses) / max(1, len(losses))
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', correct / total, epoch)

        print("\tValidation: Loss={:.2f}\t Accuracy={:.2f}\t".format(val_loss, correct / total))
        # Evaluation Loop End

        # Update "best.pth" model if val_loss in current epoch is lower than the best validation loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(out_path, "best.pth")
            )

            # Save model based on the frequency defined by "args.save_after"
        if (epoch + 1) % save_after == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(out_path, "epoch_{}.pth".format(epoch + 1))
            )

    # need to run the predict.py code to test the accuracy of the model on the test data
    prediction = PredictData(test_loader)
    prediction.predict()
