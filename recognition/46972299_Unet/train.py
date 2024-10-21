"""
Contains the code for training, validating, testing, and saving the Unet

@author Carl Flottmann
"""
from modules import Improved3DUnet
from metrics import DiceLoss
from utils import cur_time
from dataset import *
import torch
import torch.optim as optim
import time
import os
import math

# rangpur or local machine
LOCAL = True
OUTPUT_DIR = "model"

# hyperparameters
BATCH_SIZE = 1
EPOCHS = 2
NUM_CLASSES = 6  # as per powerpoint slides
INPUT_CHANNELS = 1  # greyscale
NUM_LOADED = 5  # set to None to load all
SHUFFLE = False
WORKERS = 0

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
        data_loader = ProstateLoader(LOCAL_DATA_DIR + SEMANTIC_MRS + WINDOWS_SEP,
                                     LOCAL_DATA_DIR + SEMANTIC_LABELS + WINDOWS_SEP, NUM_CLASSES, num_load=NUM_LOADED, start_t=script_start_t, batch_size=BATCH_SIZE,
                                     shuffle=SHUFFLE, num_workers=WORKERS)
        sep = WINDOWS_SEP
    else:  # on rangpur
        data_loader = ProstateLoader(RANGPUR_DATA_DIR + SEMANTIC_MRS + LINUX_SEP,
                                     RANGPUR_DATA_DIR + SEMANTIC_LABELS + LINUX_SEP, NUM_CLASSES, num_load=NUM_LOADED, start_t=script_start_t, batch_size=BATCH_SIZE,
                                     shuffle=SHUFFLE, num_workers=WORKERS)
        sep = LINUX_SEP

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
    criterion = DiceLoss(NUM_CLASSES, device)

    model = model.to(device)

    # ================== training procedure
    # output about every 10% increment
    output_epochs = int(math.ceil(0.1 * EPOCHS))
    num_digits = len(str(EPOCHS))

    model.train()
    print(f"[{cur_time(script_start_t)}] Training...")

    for epoch in range(EPOCHS):
        print(f"[{cur_time(script_start_t)}] beginning epoch {epoch}")
        for step, (image, mask) in enumerate(data_loader.train()):
            image = image.to(device)
            mask = mask.to(device)

            output = model(image)

            total_loss, class_loss = criterion(output, mask)

            optimiser.zero_grad()
            total_loss.backward()
            optimiser.step()

            print(f"[{cur_time(script_start_t)}] iteration {step} complete, with total loss: {
                  total_loss.item()} and class loss {[loss.item() for loss in class_loss]}")

        scheduler.step()
        criterion.save_epoch()

        if epoch % output_epochs == 0:
            torch.save(model.state_dict(), f".{sep}{output_dir}{
                sep}model{epoch:0{num_digits}d}.pt")

    print(f"[{cur_time(script_start_t)}] Training complete")

    criterion.save_loss_figures(f".{sep}{output_dir}{sep}")


if LOCAL:
    if __name__ == "__main__":
        main()
else:
    main()
