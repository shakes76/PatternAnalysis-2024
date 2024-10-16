import os
from pathlib import Path
import numpy as np
import nibabel as nib
from skimage import io


def to_uint8(data):
    data -= data.min()
    data /= data.max()
    data *= 255
    return data.astype(np.uint8)


def nii_to_jpgs(input_path, output_dir, rgb=False):
    # Source: https://stackoverflow.com/questions/68691070/how-to-handle-image-with-extension-nii-gz-is-it-possible-to-convert-them-in-gr
    output_dir = Path(output_dir)
    data = nib.load(input_path).get_fdata()
    *_, num_slices, num_channels = data.shape
    for channel in range(num_channels):
        volume = data[..., channel]
        volume = to_uint8(volume)
        channel_dir = output_dir / f'channel_{channel}'
        channel_dir.mkdir(exist_ok=True, parents=True)
        for slice in range(num_slices):
            slice_data = volume[..., slice]
            if rgb:
                slice_data = np.stack(3 * [slice_data], axis=2)
            output_path = channel_dir / f'channel_{channel}_slice_{slice}.jpg'
            io.imsave(output_path, slice_data)

def conv_dir(image_folder, output_dir, rbg):
    image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith('.gz')])
    for file in image_filenames:
        print(file)
        nii_to_jpgs(file, output_dir, rbg)

conv_dir(
    "/home/ja/Documents/uni/COMP3710/PatternAnalysis-2024/data/keras_slices_test",
    "/home/ja/Documents/uni/COMP3710/PatternAnalysis-2024/data_jpg/keras_slices_test",
    True
)