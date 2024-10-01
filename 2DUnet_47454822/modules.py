# Contains components of model.

from keras import layers, models

FULL_SIZE_IMG = 1  # set to 2 to use full size image
INPUT_SHAPE = (32, 32, 3)
num_classes = 4  # numb of classes in segmentation

def unet_model(input_size=(128*FULL_SIZE_IMG, 128*FULL_SIZE_IMG, 1)):
    inputs = layers.Input(input_size)

    # Encoder
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
    conv4 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv4 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv4)

    # Decoder
    up1 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv4)
    concat1 = layers.concatenate([up1, conv3], axis=3)
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(concat1)
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up2 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    concat2 = layers.concatenate([up2, conv2], axis=3)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(concat2)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    concat3 = layers.concatenate([up3, conv1], axis=3)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat3)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    concat4 = layers.concatenate([up4, conv1], axis=3)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat4)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    outputs = layers.Conv2D(4, (1, 1), activation='sigmoid')(conv9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model