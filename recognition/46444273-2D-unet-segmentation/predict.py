from keras.models import load_model
from dataset import get_training_data, load_data_2D
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from modules import dice_loss
import pathlib
import sys
from util import create_mask
import random

def main(args):
    mode = args[1]
    if mode not in ['--train', '--val', '--test']:
       print('Please choose a dataset to sample from.')
       return
    idx = random.randint(0, 100)
    if len(args) == 3:
        try:
            idx = int(args[2])
        except ValueError as verr:
           print('Please choose an integer within the chosen dataset range.')
           return
        
    model = load_model('models/model.h5', custom_objects={'dice_loss': dice_loss})

    train_data_dir = pathlib.Path('dataset/keras_slices_train').with_suffix('').glob('*.nii')
    train_seg_data_dir = pathlib.Path('dataset/keras_slices_seg_train').with_suffix('').glob('*.nii')

    image = load_data_2D([list(train_data_dir)[idx]]).reshape(256, 128, 1)
    mask = load_data_2D([list(train_seg_data_dir)[idx]]).reshape(256, 128, 1)

    print(f'image shape = {image.shape}')
    print(f'mask shape = {mask.shape}')

    from keras.utils import to_categorical
    mask = to_categorical(mask, num_classes=6, dtype=np.uint8)
    mask = mask.reshape((mask.shape[0], mask.shape[1], 6))

    test_img = image
    ground_truth = mask
    test_img_norm = test_img[:,:,:][:,:,:]
    test_img_input = np.expand_dims(test_img_norm, 0)

    prediction = (model.predict(test_img_input))

    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Original image')
    plt.imshow(test_img[:,:,0])
    plt.subplot(232)
    plt.title('Manual segmentation labels')
    ground_truth = np.expand_dims(ground_truth, 0)
    print(f'ground truth classes = {np.unique(create_mask(ground_truth))}')
    plt.imshow(create_mask(ground_truth))
    plt.subplot(233)
    plt.title('U-Net segmentation labels')
    print(f'prediction classes = {np.unique(create_mask(prediction))}')
    plt.imshow(create_mask(prediction))
    plt.savefig('plots/predict.png')
    plt.show()

if __name__ == '__main__':
   main(sys.argv)
