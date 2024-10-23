import random
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from dataset import load_data_2D
from modules import dice_coef_prostate, total_loss
from util import create_mask
import paths

def main(args):
    d_flag, mode = args[1], args[2]
    if d_flag != '-d':
        print('Please select a dataset with the flag -d <train/validate/test>.')
        return
    if mode not in {'train', 'validate', 'val', 'test'}:
       print('Please choose a dataset from train/validate/test.')
       return
    
    if mode == 'train':
        image_files = pathlib.Path(paths.TRAIN_IMG_PATH).with_suffix('').glob('*.nii')
        label_files = pathlib.Path(paths.TRAIN_LABEL_PATH).with_suffix('').glob('*.nii')
    elif mode in {'validate', 'val'}:
        image_files = pathlib.Path(paths.VAL_IMG_PATH).with_suffix('').glob('*.nii')
        label_files = pathlib.Path(paths.VAL_LABEL_PATH).with_suffix('').glob('*.nii')
    else:
        image_files = pathlib.Path(paths.TEST_IMG_PATH).with_suffix('').glob('*.nii')
        label_files = pathlib.Path(paths.TEST_LABEL_PATH).with_suffix('').glob('*.nii')

    image_data = list(image_files)
    label_data = list(label_files)

    idx = random.randint(0, len(image_data)-1)
    if len(args) == 4:
        try:
            idx = int(args[3])
        except ValueError as verr:
            print('Please choose a valid index.')
            return
        if idx < 0 or idx >= len(image_data):
            print('Index out of dataset range.')
            return

    model = load_model('models/model.h5',
                        custom_objects={
                                'total_loss': total_loss,
                                'dice_coef_prostate': dice_coef_prostate
                            })

    image = load_data_2D([image_data[idx]]).reshape(256, 128, 1)
    mask = load_data_2D([label_data[idx]]).reshape(256, 128, 1)

    from keras.utils import to_categorical
    mask = to_categorical(mask, num_classes=6, dtype=np.uint8)
    mask = mask.reshape((mask.shape[0], mask.shape[1], 6))

    test_img = image
    ground_truth = mask
    print(f'image shape = {test_img.shape}')
    test_img_norm = test_img[:,:,:][:,:,:]
    print(f'norm shape = {test_img_norm.shape}')
    test_img_input = np.expand_dims(test_img_norm, 0)

    prediction = (model.predict(test_img_input))

    from matplotlib import colors
    norm = colors.Normalize(vmin=0, vmax=5)

    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Original image')
    plt.imshow(test_img[:,:,0])
    plt.subplot(232)
    plt.title('Manual segmentation labels')
    ground_truth = np.expand_dims(ground_truth, 0)
    print(f'ground truth classes = {np.unique(create_mask(ground_truth))}')
    plt.imshow(create_mask(ground_truth), cmap='viridis', norm=norm)
    plt.subplot(233)
    plt.title('U-Net segmentation labels')
    print(f'prediction classes = {np.unique(create_mask(prediction))}')
    plt.imshow(create_mask(prediction), cmap='viridis', norm=norm)
    plt.savefig('plots/predict.png')
    plt.show()

if __name__ == '__main__':
   main(sys.argv)
