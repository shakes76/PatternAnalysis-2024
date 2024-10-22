"""
The test driver script, runs the Unet and saves the output

@author Carl Flottmann
"""
import torch
from utils import cur_time, ModelFile, save_loss_figures, ModelState
from modules import Improved3DUnet
from metrics import DiceLoss, Accuracy
from dataset import *
import time
import matplotlib.pyplot as plt

LOCAL = True
LOAD_FILE_PATH = ".\\model15\\validated_model.pt"
# make sure to include the directory separator as a suffix here
OUTPUT_PATH = ".\\model15\\"
BATCH_SIZE = 1
SHUFFLE = False
WORKERS = 0


def main() -> None:
    script_start_t = time.time()
    print(f"[{cur_time(script_start_t)}] Running prediction file for improved 3D Unet")
    # setup the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f"[{cur_time(script_start_t)}] Warning! CUDA not found, using CPU")
    else:
        print(f"[{cur_time(script_start_t)}] Running on device {device}")
    torch.autograd.set_detect_anomaly(True)  # help detect problems

    torch.serialization.add_safe_globals([ModelState])
    file_dict = torch.load(LOAD_FILE_PATH, weights_only=True)

    old_criterion = DiceLoss.load_state_dict(
        file_dict[ModelFile.CRITERION.value])
    if file_dict[ModelFile.TRAINED_LOCALLY.value]:
        print(f"[{cur_time(script_start_t)}] This model was trained locally")
    else:
        print(f"[{cur_time(script_start_t)}] This model was trained on rangpur")

    if file_dict[ModelFile.TRAINED_LOCALLY.value] != LOCAL:
        print(f"[{cur_time(script_start_t)
                  }] Warning! Not running on the same computer this model was trained on")

    # check if the model was stopped prematurely and plot the loss if it was
    if file_dict[ModelFile.STATE.value] == ModelState.TRAINING:
        print(
            f"[{cur_time(script_start_t)}] Warning! This model did not finish training")
        try:
            save_loss_figures(old_criterion, OUTPUT_PATH, "training")
        except AttributeError:
            print(
                f"[{cur_time(script_start_t)}] Warning! model had no data in criterion")

    elif file_dict[ModelFile.STATE.value] == ModelState.VALIDATING:
        print(
            f"[{cur_time(script_start_t)}] Warning! This model did not finish validating")
        try:
            save_loss_figures(old_criterion, OUTPUT_PATH, "validation")
        except AttributeError:
            print(
                f"[{cur_time(script_start_t)}] Warning! model had no data in criterion")

    else:  # state is DONE
        print(
            f"[{cur_time(script_start_t)}] This model completed training and validation")

     # get the model with the parameters back
    num_classes = file_dict[ModelFile.NUM_CLASSES.value]
    input_channels = file_dict[ModelFile.INPUT_CHANNELS.value]
    model = Improved3DUnet(input_channels, num_classes)
    model.load_state_dict(file_dict[ModelFile.MODEL.value])
    model.to(device)

    # get the data loader with the parameters back
    old_data_loader = ProstateLoader.load_state_dict(
        file_dict[ModelFile.DATA_LOADER.value])
    # setup data
    if LOCAL:
        data_loader = ProstateLoader(LOCAL_DATA_DIR + SEMANTIC_MRS + WINDOWS_SEP,
                                     LOCAL_DATA_DIR + SEMANTIC_LABELS + WINDOWS_SEP, num_load=old_data_loader.get_num_load(), start_t=script_start_t, batch_size=BATCH_SIZE,
                                     shuffle=SHUFFLE, num_workers=WORKERS)
    else:  # on rangpur
        data_loader = ProstateLoader(RANGPUR_DATA_DIR + SEMANTIC_MRS + LINUX_SEP,
                                     RANGPUR_DATA_DIR + SEMANTIC_LABELS + LINUX_SEP, num_load=old_data_loader.get_num_load(), start_t=script_start_t, batch_size=BATCH_SIZE,
                                     shuffle=SHUFFLE, num_workers=WORKERS)

    # reinitialise criterion with new device but old smooth factor for accuracy
    criterion = DiceLoss(num_classes, device, old_criterion.get_smooth())

    print(f"[{cur_time(script_start_t)}] Testing...")
    model.eval()
    accuracy = Accuracy()

    with torch.no_grad():
        for step, (image, mask) in enumerate(data_loader.test()):
            image = image.to(device)
            mask = mask.to(device)

            output = model(image)

            accuracy.forward(output, mask)
            total_loss, class_loss = criterion(output, mask)

            print(f"[{cur_time(script_start_t)}] iteration {step} complete, with total loss: {
                  total_loss.item()} and class loss {[loss.item() for loss in class_loss]}")

        print(f"[{cur_time(script_start_t)}] Test accuracy: {
              accuracy.accuracy()}%")

    print(f"[{cur_time(script_start_t)}] Testing complete")
    save_loss_figures(criterion, OUTPUT_PATH, "testing")


if LOCAL:
    if __name__ == "__main__":
        main()
else:
    main()
