import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, \
    BatchNormalization, ReLU, Conv2DTranspose, Concatenate, Cropping2D

class UNetSegmentation():

    def __init__(self):
        # define the models architecture

        inputs = Input(shape=(256, 128, 1))  # Added channel dimension
        # downsampling layers
        down1 = Conv2D(64, (3, 3), padding='same')(inputs) #(64, 254, 126)
        down1 = BatchNormalization()(down1) 
        down1 = ReLU()(down1) 
        down1 = Conv2D(64, (3, 3), padding='same')(down1) #(64, 252, 124)
        down1 = BatchNormalization()(down1) 
        down1 = ReLU()(down1) 
        pool1 = MaxPooling2D((2, 2))(down1) #(64, 126, 63)

        down2 = Conv2D(128, (3, 3), padding='same')(pool1) #(128, 124, 62)
        down2 = BatchNormalization()(down2)
        down2 = ReLU()(down2)
        down2 = Conv2D(128, (3, 3), padding='same')(down2) #(128, 122, 60)
        down2 = BatchNormalization()(down2)
        down2 = ReLU()(down2)
        pool2 = MaxPooling2D((2, 2))(down2) #(128, 62, 31)

        down3 = Conv2D(256, (3, 3), padding='same')(pool2) #(256, 60, 30)    
        down3 = BatchNormalization()(down3)
        down3 = ReLU()(down3)
        down3 = Conv2D(256, (3, 3), padding='same')(down3) #(256, 58, 28)
        down3 = BatchNormalization()(down3)
        down3 = ReLU()(down3)
        pool3 = MaxPooling2D((2, 2))(down3) #(256, 30, 15)

        down4 = Conv2D(512, (3, 3), padding='same')(pool3) #(512, 28, 14)    
        down4 = BatchNormalization()(down4)
        down4 = ReLU()(down4)
        down4 = Conv2D(512, (3, 3), padding='same')(down4) #(512, 26, 12)
        down4 = BatchNormalization()(down4)
        down4 = ReLU()(down4)
        pool4 = MaxPooling2D((2, 2))(down4) #(512, 14, 7)

        # upsampling layers
        up1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(pool4)  # (256, 28, 14)
        crop1 = tf.keras.layers.Cropping2D(cropping=((2, 2), (0, 0)))(down3)  # Crop down3 to (256, 28, 14)
        concat1 = Concatenate()([up1, crop1])
        upConv1 = Conv2D(256, (3, 3), padding='same')(concat1)
        upConv1 = BatchNormalization()(upConv1)
        upConv1 = ReLU()(upConv1)  
        upConv1 = Conv2D(256, (3, 3), padding='same')(upConv1)
        upConv1 = BatchNormalization()(upConv1)
        upConv1 = ReLU()(upConv1)   

        up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(upConv1)  # (128, 56, 28)
        crop2 = tf.keras.layers.Cropping2D(cropping=((4, 4), (2, 2)))(down2)  # Crop down2 to (128, 56, 28)
        concat2 = Concatenate()([up2, crop2])
        upConv2 = Conv2D(128, (3, 3), padding='same')(concat2)
        upConv2 = BatchNormalization()(upConv2)
        upConv2 = ReLU()(upConv2)  
        upConv2 = Conv2D(128, (3, 3), padding='same')(upConv2)
        upConv2 = BatchNormalization()(upConv2)
        upConv2 = ReLU()(upConv2)

        up3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(upConv2)  # (64, 112, 56)
        crop3 = tf.keras.layers.Cropping2D(cropping=((8, 8), (4, 4)))(down1)  # Crop down1 to (64, 112, 56)
        concat3 = Concatenate()([up3, crop3])
        upConv3 = Conv2D(64, (3, 3), padding='same')(concat3)
        upConv3 = BatchNormalization()(upConv3)
        upConv3 = ReLU()(upConv3)
        upConv3 = Conv2D(64, (3, 3), padding='same')(upConv3)
        upConv3 = BatchNormalization()(upConv3)
        upConv3 = ReLU()(upConv3)

        # output layer
        output = Conv2D(1, (1, 1), padding='same')(upConv3)

