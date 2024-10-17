"""
Contains the code for training, validating, testing, and saving the Unet

@author Carl Flottmann
"""
from modules import Improved3DUnet
from metrics import get_loss_function
from utils import cur_time
from dataset import *
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import time
import os
import math

# rangpur or local machine
LOCAL = False
OUTPUT_DIR = "model"

# hyperparameters
BATCH_SIZE = 1
EPOCHS = 2
NUM_CLASSES = 6  # as per powerpoint slides
INPUT_CHANNELS = 1  # greyscale
NUM_LOADED = 5  # set to None to load all
SHUFFLE = False
WORKERS = 1

# taken from the paper on the improved unet
INITIAL_LR = 5e-4
WEIGHT_DECAY = 1e-5
DECAY_FACTOR = 0.985


def main() -> None:
    script_start_t = time.time()
    print(f"[{cur_time(script_start_t)}] Running training file for improved 3D Unet")
    # setup the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f"[{cur_time(script_start_t)}] Warning! CUDA not found, using CPU")
    else:
        print(f"[{cur_time(script_start_t)}] Running on device {device}")
    torch.autograd.set_detect_anomaly(True)  # help detect problems

    # setup data
    if LOCAL:
        dataset = ProstateDataset(LOCAL_DATA_DIR + SEMANTIC_MRS + WINDOWS_SEP,
                                  LOCAL_DATA_DIR + SEMANTIC_LABELS + WINDOWS_SEP, NUM_CLASSES, num_load=NUM_LOADED)
        sep = WINDOWS_SEP
    else:  # on rangpur
        dataset = ProstateDataset(RANGPUR_DATA_DIR + SEMANTIC_MRS + LINUX_SEP,
                                  RANGPUR_DATA_DIR + SEMANTIC_LABELS + LINUX_SEP, NUM_CLASSES, num_load=NUM_LOADED)
        sep = LINUX_SEP

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                             shuffle=SHUFFLE, num_workers=WORKERS)

    # create output directory
    output_dir = OUTPUT_DIR
    dir_name_count = 1

    while os.path.exists(f"{os.getcwd()}{sep}{output_dir}"):
        output_dir = f"{OUTPUT_DIR}{dir_name_count}"
        dir_name_count += 1
    os.makedirs(f"{os.getcwd()}{sep}{output_dir}")

    print(f"[{cur_time(script_start_t)}] will output all relevent files to {
        output_dir} in running directory")

    # setup model
    model = Improved3DUnet(INPUT_CHANNELS, NUM_CLASSES)
    # optimiser taken from the paper on the improved unet
    optimiser = optim.Adam(model.parameters(), lr=INITIAL_LR,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimiser, lr_lambda=lambda epoch: DECAY_FACTOR ** epoch)
    # will use dice loss as per the task requirements
    criterion = get_loss_function()

    model = model.to(device)

    # ================== training procedure
    # output about every 10% increment
    output_epochs = int(math.ceil(0.1 * EPOCHS))
    num_digits = len(str(EPOCHS))

    model.train()
    print(f"[{cur_time(script_start_t)}] Training...")
    model_start_t = time.time()

    for epoch in range(EPOCHS):
        for step, (image, mask) in enumerate(data_loader):
            image = image.to(device)
            mask = mask.to(device)

            output = model(image)

            # print(f"[TRAIN] input: {image.shape}, output: {output.shape}, mask: {mask.shape}")
            loss = criterion(output, mask)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            print(f"[{cur_time(script_start_t)}] iteration {
                step} complete after {cur_time(model_start_t)} training, loss: {loss.item()}")

        scheduler.step()

        if epoch % output_epochs == 0:
            torch.save(model.state_dict(), f".{sep}{output_dir}{
                sep}model{epoch:0{num_digits}d}.pt")

    print(f"Training took {cur_time(model_start_t)} in total")


if __name__ == "__main__":
    main()
