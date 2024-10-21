"""
The test driver script, runs the Unet and saves the output

@author Carl Flottmann
"""
import torch
from utils import cur_time, ModelState
from metrics import DiceLoss
import time
import matplotlib.pyplot as plt

# rangpur or local machine
LOCAL = True
LOAD_FILE_PATH = ".\\model\\final_model.pt"
OUTPUT_PATH = ".\\model\\"


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

    output = torch.load(LOAD_FILE_PATH, weights_only=True)
    model = output[ModelState.MODEL.value]
    criterion = DiceLoss.load_state_dict(output[ModelState.CRITERION.value])

    # first do complete dice loss
    losses = criterion.get_all_losses()

    x_axis = list(range(len(losses[0])))

    plt.plot(x_axis, losses[0], label="Total Loss", marker='o')
    for i, class_loss in enumerate(losses[1:]):
        plt.plot(x_axis, class_loss, label=f"Class {
            i + 1} Loss", marker='o')

    plt.xlabel("Total iterations (including epochs)")
    plt.ylabel("DICE loss")
    plt.title("Complete DICE Loss Over Training")
    plt.legend()
    plt.grid()
    plt.savefig(f"{OUTPUT_PATH}complete_dice_loss.png")
    plt.close()

    # second do average dice loss
    losses = criterion.get_average_losses()

    x_axis = list(range(len(losses[0])))

    plt.plot(x_axis, losses[0], label="Total Loss", marker='o')
    for i, class_loss in enumerate(losses[1:]):
        plt.plot(x_axis, class_loss, label=f"Class {
            i + 1} Loss", marker='o')

    plt.xlabel("Total epochs")
    plt.ylabel("DICE loss")
    plt.title("Average DICE Loss Over Training")
    plt.legend()
    plt.grid()
    plt.savefig(f"{OUTPUT_PATH}average_dice_loss.png")
    plt.close()

    # last do end dice loss
    losses = criterion.get_end_losses()

    x_axis = list(range(len(losses[0])))

    plt.plot(x_axis, losses[0], label="Total Loss", marker='o')
    for i, class_loss in enumerate(losses[1:]):
        plt.plot(x_axis, class_loss, label=f"Class {
            i + 1} Loss", marker='o')

    plt.xlabel("Total epochs")
    plt.ylabel("DICE loss")
    plt.title("DICE Loss at the End of Each Epoch Over Training")
    plt.legend()
    plt.grid()
    plt.savefig(f"{OUTPUT_PATH}end_dice_loss.png")
    plt.close()


if LOCAL:
    if __name__ == "__main__":
        main()
else:
    main()
