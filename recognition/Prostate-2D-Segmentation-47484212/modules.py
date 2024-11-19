import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, \
    BatchNormalization, ReLU, Conv2DTranspose, Concatenate
import numpy as np

class UNetSegmentation():

    def __init__(self, modelPath=None):
        '''
        tries to load model at modelPath, or builds new model
        '''
        if modelPath is not None:
            try:
                self.model = tf.keras.models.load_model(modelPath, custom_objects={'dice_loss':dice_loss})
            except Exception:
                print(f"model at {modelPath} could not be loaded, initializing new model")
                self.model = self.build()
        else:
            self.model = self.build()
        self.model.compile(optimizer='adam', loss=dice_loss) 

    def build(self):
        '''
        sets up layers and returns a model containing them
        '''
        # define the models architecture

        inputs = Input(shape=(256, 128, 1))  # Added channel dimension
        # downsampling layers
        down1 = Conv2D(64, (3, 3), padding='same')(inputs) #(256, 128, 64)
        down1 = BatchNormalization()(down1) 
        down1 = ReLU()(down1) 
        down1 = Conv2D(64, (3, 3), padding='same')(down1) 
        down1 = BatchNormalization()(down1) 
        down1 = ReLU()(down1) 
        pool1 = MaxPooling2D((2, 2))(down1) #(128, 64, 64)

        down2 = Conv2D(128, (3, 3), padding='same')(pool1) #(128, 64, 128)
        down2 = BatchNormalization()(down2)
        down2 = ReLU()(down2)
        down2 = Conv2D(128, (3, 3), padding='same')(down2) 
        down2 = BatchNormalization()(down2)
        down2 = ReLU()(down2)
        pool2 = MaxPooling2D((2, 2))(down2) #(64, 32, 128)

        down3 = Conv2D(256, (3, 3), padding='same')(pool2) #(64, 32, 256)    
        down3 = BatchNormalization()(down3)
        down3 = ReLU()(down3)
        down3 = Conv2D(256, (3, 3), padding='same')(down3) 
        down3 = BatchNormalization()(down3)
        down3 = ReLU()(down3)
        pool3 = MaxPooling2D((2, 2))(down3) #(32, 16, 256)

        down4 = Conv2D(512, (3, 3), padding='same')(pool3) #(32, 16, 512)    
        down4 = BatchNormalization()(down4)
        down4 = ReLU()(down4)
        down4 = Conv2D(512, (3, 3), padding='same')(down4) 
        down4 = BatchNormalization()(down4)
        down4 = ReLU()(down4)

        # upsampling layers
        up1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(down4)  # (64, 32, 256)
        concat1 = Concatenate()([up1, down3])
        upConv1 = Conv2D(256, (3, 3), padding='same')(concat1)
        upConv1 = BatchNormalization()(upConv1)
        upConv1 = ReLU()(upConv1)  
        upConv1 = Conv2D(256, (3, 3), padding='same')(upConv1)
        upConv1 = BatchNormalization()(upConv1)
        upConv1 = ReLU()(upConv1)   

        up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(upConv1)  # (128, 64, 128)
        concat2 = Concatenate()([up2, down2])
        upConv2 = Conv2D(128, (3, 3), padding='same')(concat2)
        upConv2 = BatchNormalization()(upConv2)
        upConv2 = ReLU()(upConv2)  
        upConv2 = Conv2D(128, (3, 3), padding='same')(upConv2)
        upConv2 = BatchNormalization()(upConv2)
        upConv2 = ReLU()(upConv2)

        up3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(upConv2)  # (256, 128, 64)
        concat3 = Concatenate()([up3, down1])
        upConv3 = Conv2D(64, (3, 3), padding='same')(concat3)
        upConv3 = BatchNormalization()(upConv3)
        upConv3 = ReLU()(upConv3)
        upConv3 = Conv2D(64, (3, 3), padding='same')(upConv3)
        upConv3 = BatchNormalization()(upConv3)
        upConv3 = ReLU()(upConv3)

        # output layer
        output = Conv2D(5, (1, 1), padding='same', activation='softmax')(upConv3)

        model = tf.keras.Model(inputs=inputs, outputs=output)
        return model

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

def dice_loss(y_true, y_pred):
    '''
    my implementation of dice loss
    '''
    classLosses = []
    for i in range(5):
        PredMasks = y_pred[:, :, :, i]
        RealMasks = y_true[:, :, :, i]
        classLosses.append(1 -  (2 * tf.reduce_sum(PredMasks * RealMasks) + 1e-9) / (tf.reduce_sum(PredMasks + RealMasks) + 1e-9))
    return tf.reduce_mean(classLosses)
