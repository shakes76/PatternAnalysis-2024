"""
Contains helper miscellaneous functions used

@author Carl Flottmann
"""

import numpy as np
import nibabel as nib
from tqdm import tqdm
import time
from enum import Enum
import matplotlib.pyplot as plt
from metrics import DiceLoss


class ModelState(Enum):
    TRAINING = 0
    VALIDATING = 1
    DONE = 2


class ModelFile(Enum):
    MODEL = "model"
    CRITERION = "criterion"
    INPUT_CHANNELS = "input_channels"
    NUM_CLASSES = "num_classes"
    DATA_LOADER = "data_loader"
    TRAINED_LOCALLY = "trained_locally"
    STATE = "state"


def save_loss_figures(criterion: DiceLoss, output_path: str, mode: str, file_names: tuple[str, str, str] = ("complete_dice", "average_dice", "end_dice")) -> None:
    # first do complete dice loss
    losses = criterion.get_all_losses()
    x_axis = list(range(len(losses[0])))
    plt.plot(x_axis, losses[0], label="Total Loss")
    for i, class_loss in enumerate(losses[1:]):
        plt.plot(x_axis, class_loss, label=f"Class {i + 1} Loss")
    plt.xlabel("Total iterations (including epochs)")
    plt.ylabel("DICE loss")
    plt.title(f"Complete DICE Loss over {mode}")
    size = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(size[0] * 2, size[1] * 2)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.grid()
    plt.savefig(f"{output_path}{file_names[0]}_{mode}.png")
    plt.close()

    # second do average dice loss
    losses = criterion.get_average_losses()
    x_axis = list(range(len(losses[0])))
    plt.plot(x_axis, losses[0], label="Total Loss")
    for i, class_loss in enumerate(losses[1:]):
        plt.plot(x_axis, class_loss, label=f"Class {i + 1} Loss")
    plt.xlabel("Total epochs")
    plt.ylabel("DICE loss")
    plt.title(f"Average DICE Loss over {mode}")
    size = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(size[0] * 2, size[1] * 2)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.grid()
    plt.savefig(f"{output_path}{file_names[1]}_{mode}.png")
    plt.close()

    # last do end dice loss
    losses = criterion.get_end_losses()
    x_axis = list(range(len(losses[0])))
    plt.plot(x_axis, losses[0], label="Total Loss")
    for i, class_loss in enumerate(losses[1:]):
        plt.plot(x_axis, class_loss, label=f"Class {i + 1} Loss")
    plt.xlabel("Total epochs")
    plt.ylabel("DICE loss")
    plt.title(f"DICE Loss at the end of each epoch ever {mode}")
    size = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(size[0] * 2, size[1] * 2)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.grid()
    plt.savefig(f"{output_path}{file_names[2]}_{mode}.png")
    plt.close()


def cur_time(start: float) -> float:
    elapsed_time = time.time() - start

    if elapsed_time < 1:  # display in milliseconds
        return f"{elapsed_time * 1000:.2f}ms"

    elif elapsed_time < 60:  # display in seconds
        seconds = int(elapsed_time)
        milliseconds = (elapsed_time - seconds) * 1000
        return f"{seconds}s {milliseconds:.2f}ms"

    elif elapsed_time < 3600:  # display in minutes
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        return f"{minutes}min {seconds}s"

    else:  # display in hours
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        return f"{hours}hrs {minutes}min"


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:(c + 1)][arr == c] = 1
    return res

# load medical image functions


def load_data_2D(imageNames: list[str], normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False) -> np.ndarray:
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0 -1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')

    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            # sometimes extra dims in HipMRI_study data
            inImage = inImage[:, :, 0]

        inImage = inImage.astype(dtype)

        if normImage:
            # ~ inImage = inImage / np.linalg.norm(inImage)
            # ~ inImage = 255. * inImage / inImage.max ()
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            # inImage = utils.to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


def load_data_3D(imageNames: list[str], normImage=False, categorical=False, dtype=np.float32, getAffines=False, orient=False, early_stop=False) -> np.ndarray:
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0 -1.0)
    orient: Apply orientation and resample image? Good for images with large slice thickness or anisotropic resolution
    dtype: Type of the data. If dtype=np.uint8, it is assumed that the data is labels
    early_stop: Stop loading pre-maturely? Leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # ~ interp = 'continuous'
    interp = 'linear'
    if dtype == np.uint8:  # assume labels
        interp = 'nearest'

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])

    if orient:
        # niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
        pass
        # ~ testResultName = "oriented.nii.gz"
        # ~ niftiImage.to_filename(testResultName)

    first_case = niftiImage.get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims , remove

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if orient:
            # niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
            pass

        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            # sometimes extra dims in HipMRI_study data
            inImage = inImage[:, :, :, 0]

        inImage = inImage[:, :, :depth]  # clip slices
        inImage = inImage.astype(dtype)

        if normImage:
            # ~ inImage = inImage / np.linalg.norm(inImage)
            # ~ inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            # inImage = utils.to_channels(inImage, dtype=dtype)
            # ~ images [i, :, :, :, :] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1],
                   :inImage.shape[2], :inImage.shape[3]] = inImage  # with pad
        else:
            # ~ images [i, :, :, :] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1],
                   :inImage.shape[2]] = inImage  # with pad

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images
