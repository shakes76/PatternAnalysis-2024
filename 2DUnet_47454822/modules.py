# Contains components of model.
import keras
from keras import layers, models, Sequential
from keras.src.backend import shape
from keras.src.losses import Dice
from keras.src.optimizers import AdamW, Adam
from keras.src import backend
from keras.src import ops
# from keras_core.src.ops import argmax
from tensorflow.python.keras.losses import CategoricalCrossentropy



# Hyperparams
FULL_SIZE_IMG = 1  # set to 2 to use full size image
INPUT_SHAPE = (32, 32, 3)
num_classes = 4  # numb of classes in segmentation


# The loss function we use when evaluating
def dice_loss(y_true, y_pred, axis=None):
    # this is the Dice() code
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.cast(y_true, y_pred.dtype)

    inputs = y_true
    targets = y_pred

    intersection = ops.sum(inputs * targets, axis=axis)
    dice = ops.divide(
        2.0 * intersection,
        ops.sum(y_true, axis=axis)
        + ops.sum(y_pred, axis=axis)
        + backend.epsilon(),
    )

    return 1 - dice


# the main model
def unet_model(input_size=(128, 128, 1), batch_size=12, preprocessing=None):
    keras.backend.clear_session()

    inputs = layers.Input(input_size)

    # A basic convolution and pool layer. we have 4 of these that process higher and higher level features then pass on their findings to the next.
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)  # padding same accounts for the shrinkage that occurs from kernal
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)


    # Bridge
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    # Decoder. Here we're up scaling the previous output, concatenating it with the corresponding Encoder layer, then convolving it and passing it on.
    up1 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    concat1 = layers.concatenate([up1, conv4], axis=3)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(concat1)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up2 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    concat2 = layers.concatenate([up2, conv3], axis=3)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(concat2)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    concat3 = layers.concatenate([up3, conv2], axis=3)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat3)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    concat4 = layers.concatenate([up4, conv1], axis=3)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat4)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    outputs = layers.Conv2D(6, (1, 1), activation='softmax')(conv9)

    # Bring everything together and actually create the model object
    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=AdamW(), loss="categorical_crossentropy", metrics=['accuracy'])

    return model
