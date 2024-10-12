import os

import keras
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import AdamW
from keras.src.saving.saving_api import load_model
from keras.src.utils import to_categorical

from dataset import load_dir, load_data_2D
from modules import unet_model
from keras.src.losses import Dice

# [BATCH_SIZE, CHANNELS, HEIGHT, WIDTH]
# load_dir return: [BATCH_SIZE, HEIGHT, WIDTH]

def test_cuda():
    import tensorflow as tf
    print(tf.config.list_physical_devices('GPU'))

def invoke():
        result = load_data_2D(["/home/ja/Documents/uni/COMP3710/PatternAnalysis-2024/data/keras_slices_seg_train/seg_004_week_0_slice_0.nii.gz",
                               "/home/ja/Documents/uni/COMP3710/PatternAnalysis-2024/data/keras_slices_seg_train/seg_004_week_0_slice_1.nii.gz"])
        print(result)

def invoke_dir():
    X = load_dir("/home/ja/Documents/uni/COMP3710/PatternAnalysis-2024/data/keras_slices_test/")

# def test_experimental():
#
#     train_batches = image.ImageDataGenerator(
#         preprocessing_function=keras.applications.vgg16.preprocess_input,
#         horizontal_flip=True,
#         rescale=1./255
#     ).flow_from_directory(
#         directory="/home/ja/Documents/uni/COMP3710/PatternAnalysis-2024/data",
#         target_size=(224, 224),
#         batch_size=10,
#         shuffle=True,
#     )
#
#     print(train_batches.filenames)

def test_train():

    path = "/home/ja/Documents/uni/COMP3710/PatternAnalysis-2024/data/"

    # ============== Train or Load Model ==============

    # datagen = ImageDataGenerator(
    #     rotation_range=10,  # rotation
    #     width_shift_range=0.2,  # horizontal shift
    #     height_shift_range=0.2,  # vertical shift
    #     zoom_range=0.2,  # zoom
    #     horizontal_flip=True,  # horizontal flip
    #     brightness_range=[0.2, 1.2])  # brightness
    #
    # train_generator = datagen.flow_from_directory(
    #
    #     directory=f"{path}keras_slices_train/",
    #     # target_size=(400, 400),  # resize to this size
    #     # color_mode="rgb",  # for coloured images
    #     batch_size=1,  # number of images to extract from folder for every batch
    #     classes=6,  # classes to predict
    #     seed=2020  # to make the result reproducible
    # )


    train_images = load_dir(f"{path}keras_slices_train/")
    train_masks = load_dir(f"{path}keras_slices_seg_train/")

    test_images = load_dir(f"{path}keras_slices_test/")
    test_masks = load_dir(f"{path}keras_slices_seg_test/")
    # train_images = test_images
    # train_masks = test_masks


    model_name = "drop_norm.crop.flip_train13"

    if os.path.isfile(f'models/{model_name}.keras') is False:
        # https://fdnieuwveldt.medium.com/building-advanced-custom-feature-transformation-pipelines-in-keras-using-easyflow-4c5fce545dc2

        print('Training Model')


        batch_size = 32
        train_images = np.expand_dims(train_images, axis=-1)  # Add a channel dimension, making the shape (batch_size, 256, 128, 1)
        # X = np.resize(X, (batch_size, 128, 128, 1))

        train_masks = np.expand_dims(train_masks, axis=-1)
        train_masks = to_categorical(train_masks, num_classes=6)  # Otherwise it has (None, 256, 128, 1) but we want (None, 256, 128, 6) where the final is the num channels/classes
        # y = np.squeeze(y, axis=-1)
        # y = np.resize(y, (batch_size, 128, 128, 1))

        print(f"X Shape: {np.shape(train_images)}")
        print(f"y Shape: {np.shape(train_masks)}")


        model = unet_model((256, 128, 1), batch_size=batch_size)
        # model.compile(optimizer=AdamW(), loss=Dice(), metrics=['accuracy']) # used to be sparse_cat_crossent. find better loss
        # TODO: add IoU loss. See https://keras.io/api/keras_cv/losses/iou_loss/  keras_cv.losses.IoULoss()

        model.fit(x=train_images, y=train_masks, batch_size=batch_size, epochs=1, shuffle=True, verbose=2)

        print('Finished training, saving')
        model.save(f'models/{model_name}.keras')
        print('Saved and DONE')

    else:
        test_images = np.expand_dims(test_images, axis=-1)  # Add a channel dimension, making the shape (batch_size, 256, 128, 1)
        # X = np.resize(X, (batch_size, 128, 128, 1))

        test_masks = np.expand_dims(test_masks, axis=-1)
        test_masks = to_categorical(test_masks, num_classes=6)  # Otherwise it has (None, 256, 128, 1) but we want (None, 256, 128, 6) where the final is the num channels/classes

        print("Loading Model")
        model = load_model(f'models/{model_name}.keras')

        loss, accuracy = model.evaluate(test_images, test_masks)

        print(f"Loss: {loss}")
        print(f"Accuracy: {accuracy}")

if __name__ == '__main__':
    # invoke_dir()
    test_train()
    #test_cuda()
    print("DONE")

    # minimum Dice similarity coefficient of 0.75 on the test set on the prostate label.

"""
Removed:
/home/ja/Documents/uni/COMP3710/PatternAnalysis-2024/data/keras_slices_train/case_019_week_1_slice_0.nii.gz

"""
# POL7010 Dynamics of Governance