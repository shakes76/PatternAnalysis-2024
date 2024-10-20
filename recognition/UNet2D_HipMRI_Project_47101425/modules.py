import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu


class UNet(tf.keras.Model):
    def __init__(self, input_dims, latent_dim=64, channels=1, activation='sigmoid'):
        super(UNet, self).__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.activation = activation
        self.input_dims = input_dims

        # Initialize the model
        self.model = self.UNet2D_compact()

    def UNet2D_compact(self):
        inputs = Input(shape=self.input_dims)

        # Encoder
        down1 = self.norm_conv2d(inputs, self.latent_dim // 16)
        down1 = self.norm_conv2d(down1, self.latent_dim // 16)
        pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(down1)

        down2 = self.norm_conv2d(pool1, self.latent_dim // 8)
        down2 = self.norm_conv2d(down2, self.latent_dim // 8)
        pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(down2)

        down3 = self.norm_conv2d(pool2, self.latent_dim // 4)
        down3 = self.norm_conv2d(down3, self.latent_dim // 4)
        pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(down3)

        down4 = self.norm_conv2d(pool3, self.latent_dim // 2)
        down4 = self.norm_conv2d(down4, self.latent_dim // 2)
        pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(down4)

        latent = self.norm_conv2d(pool4, self.latent_dim)
        latent = self.norm_conv2d(latent, self.latent_dim)

        # Decoder
        up4 = Conv2DTranspose(self.latent_dim // 2, kernel_size=(2, 2), strides=(2, 2), padding='same')(latent)
        up4 = Concatenate(axis=-1)([up4, down4])
        up4 = self.norm_conv2d(up4, self.latent_dim // 2)
        up4 = self.norm_conv2d(up4, self.latent_dim // 2)

        up3 = Conv2DTranspose(self.latent_dim // 4, kernel_size=(2, 2), strides=(2, 2), padding='same')(up4)
        up3 = Concatenate(axis=-1)([up3, down3])
        up3 = self.norm_conv2d(up3, self.latent_dim // 4)
        up3 = self.norm_conv2d(up3, self.latent_dim // 4)

        up2 = Conv2DTranspose(self.latent_dim // 8, kernel_size=(2, 2), strides=(2, 2), padding='same')(up3)
        up2 = Concatenate(axis=-1)([up2, down2])
        up2 = self.norm_conv2d(up2, self.latent_dim // 8)
        up2 = self.norm_conv2d(up2, self.latent_dim // 8)

        up1 = Conv2DTranspose(self.latent_dim // 16, kernel_size=(2, 2), strides=(2, 2), padding='same')(up2)
        up1 = Concatenate(axis=-1)([up1, down1])
        up1 = self.norm_conv2d(up1, self.latent_dim // 16)
        up1 = self.norm_conv2d(up1, self.latent_dim // 16)

        # Output Layer
        outputs = Conv2D(self.channels, kernel_size=(1, 1), activation=self.activation)(up1)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def norm_conv2d(self, x, filters):
        return Conv2D(filters, kernel_size=(3, 3), padding='same', activation=relu)(x)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)
