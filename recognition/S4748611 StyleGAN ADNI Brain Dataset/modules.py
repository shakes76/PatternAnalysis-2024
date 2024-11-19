import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Layer, UpSampling2D, BatchNormalization, Add, Lambda
from tensorflow.keras.models import Model
import numpy as np

# Define constants
LATENT_DIM = 512 
INITIAL_SIZE = 8 
NUM_CHANNELS = 1  
FINAL_SIZE = 128 

# Custom AdaIN (Adaptive Instance Normalization) Layer
class AdaIN(Layer):
    def __init__(self, num_channels=None, **kwargs):
        """Initialize AdaIN layer with optional channel specification."""
        super(AdaIN, self).__init__(**kwargs)
        self.num_channels = num_channels 

    def build(self, input_shape):
        """Build the AdaIN layer with style transformation weights."""
        if self.num_channels is None:
            self.num_channels = input_shape[0][-1] 
        # Learnable affine transformation for scale and shift based on style input
        self.style_scale_transform = Dense(self.num_channels)
        self.style_shift_transform = Dense(self.num_channels)

    def call(self, inputs):
        """Apply normalization and style transformations to the input."""
        x, style = inputs
        style_scale = self.style_scale_transform(style) 
        style_shift = self.style_shift_transform(style) 

        # Calculate mean and variance for the normalization of x
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        normalized = (x - mean) / tf.sqrt(variance + 1e-8)

        # Apply style-based scaling and shifting to the normalized input
        return normalized * (1 + style_scale[:, None, None, :]) + style_shift[:, None, None, :]

    def get_config(self):
        """Return the configuration of the layer for serialization."""
        config = super(AdaIN, self).get_config()
        config.update({
            "num_channels": self.num_channels,  
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create a layer instance from configuration."""
        return cls(**config) 

# Custom layer to apply noise to the input
class ApplyNoise(Layer):
    def __init__(self, channels, **kwargs):
        """Initialize the ApplyNoise layer with channel count."""
        super(ApplyNoise, self).__init__(**kwargs)
        self.channels = channels

    def build(self, input_shape):
        """Build the ApplyNoise layer by initializing noise weights."""
        # Learnable noise weight applied to the noise input
        self.noise_weight = self.add_weight(shape=(self.channels,),
                                            initializer='zeros',
                                            trainable=True)

    def call(self, inputs, **kwargs):
        """Apply noise to the input tensor."""
        noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=1.0) 
        return inputs + noise * self.noise_weight

    def get_config(self):
        """Return the configuration of the layer for serialization."""
        config = super(ApplyNoise, self).get_config()
        config.update({
            "channels": self.channels,
        })
        return config

# Mapping network used in StyleGAN to map latent vectors to intermediate space
def build_mapping_network():
    latent_input = Input(shape=(LATENT_DIM,))
    x = latent_input
    # Fully connected layers with LeakyReLU activation to map latent input to style space
    for _ in range(8):
        x = Dense(LATENT_DIM)(x)
        x = LeakyReLU(0.2)(x)
    return Model(latent_input, x, name="mapping_network")

# Synthesis network used in StyleGAN to generate images from style input and a constant input
def build_synthesis_network():
    style_input = Input(shape=(LATENT_DIM,)) 
    constant_input = Input(shape=(INITIAL_SIZE, INITIAL_SIZE, LATENT_DIM))

    x = constant_input
    
    # Initial convolution block
    x = Conv2D(512, 3, padding='same')(x)  
    x = ApplyNoise(x.shape[-1])(x) 
    x = AdaIN()([x, style_input])
    x = LeakyReLU(0.2)(x)

    # Upsampling blocks to progressively increase image resolution
    for i, filters in enumerate([512, 256, 128, 64]):
        x = UpSampling2D()(x) 
        x = Conv2D(filters, 3, padding='same')(x)  
        x = ApplyNoise(x.shape[-1])(x)  
        x = BatchNormalization()(x)  
        x = AdaIN()([x, style_input]) 
        x = LeakyReLU(0.2)(x)

        x = Conv2D(filters, 3, padding='same')(x) 
        x = ApplyNoise(x.shape[-1])(x)  
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    # Final convolution to output a single channel image
    x = Conv2D(NUM_CHANNELS, 1, activation='tanh')(x)
    
    return Model([constant_input, style_input], x, name="synthesis_network")

# Generator model that combines the mapping and synthesis networks
def build_generator():
    latent_input = Input(shape=(LATENT_DIM,))  
    constant_input = Input(shape=(INITIAL_SIZE, INITIAL_SIZE, LATENT_DIM))  

    mapping_network = build_mapping_network()  
    synthesis_network = build_synthesis_network()  

    w = mapping_network(latent_input) 
    output = synthesis_network([constant_input, w])

    return Model([latent_input, constant_input], output, name="generator")

# Discriminator model used to classify real vs fake images
def build_discriminator():
    input_image = Input(shape=(FINAL_SIZE, FINAL_SIZE, NUM_CHANNELS)) 
    
    # Convolutional layers with strided convolution for downsampling
    x = Conv2D(64, 3, strides=2, padding='same')(input_image)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, 3, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x) 
    
    # Dense layers for classification
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(1, activation='sigmoid')(x)  

    return Model(input_image, x, name="discriminator")

# StyleGAN model combining generator and discriminator for adversarial training
def build_stylegan():
    latent_input = Input(shape=(LATENT_DIM,))  
    constant_input = Input(shape=(INITIAL_SIZE, INITIAL_SIZE, LATENT_DIM)) 

    generator = build_generator() 
    discriminator = build_discriminator() 

    generated_image = generator([latent_input, constant_input])
    validity = discriminator(generated_image) 

    return Model([latent_input, constant_input], validity, name="stylegan")
