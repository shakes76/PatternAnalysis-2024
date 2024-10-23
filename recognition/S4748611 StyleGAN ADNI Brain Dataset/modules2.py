import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, LeakyReLU, Flatten, Layer, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np

# Define constants
LATENT_DIM = 512
INITIAL_SIZE = 4
NUM_CHANNELS = 1
FINAL_SIZE = 64

class WeightDemod(Layer):
    def __init__(self, filters, **kwargs):
        super(WeightDemod, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        # The style input will be transformed to match the number of filters in the convolution layer
        self.style_dense = Dense(self.filters)

    def call(self, inputs):
        x, style = inputs
        # Transform the style input to match the feature map's channel size
        style = self.style_dense(style)

        # Apply style scaling (modulation)
        weight = style[:, None, None, :] * x

        # Perform weight demodulation
        demodulation_factor = tf.math.rsqrt(tf.reduce_mean(tf.square(weight), axis=[1, 2, 3], keepdims=True) + 1e-8)
        return weight * demodulation_factor


def apply_noise(x):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=1.0)
    noise_weight = tf.Variable(initial_value=tf.zeros(shape=(x.shape[-1],)), trainable=True)
    return x + noise * noise_weight

def build_mapping_network():
    latent_input = Input(shape=(LATENT_DIM,))
    x = Dense(512)(latent_input)
    x = LeakyReLU(0.2)(x)

    for _ in range(7):  # Typically 8 fully connected layers for mapping
        x = Dense(512)(x)
        x = LeakyReLU(0.2)(x)

    return Model(latent_input, x, name="mapping_network")

def build_synthesis_network():
    style_input = Input(shape=(LATENT_DIM,))
    constant_input = Input(shape=(INITIAL_SIZE, INITIAL_SIZE, LATENT_DIM))

    x = constant_input

    # Initial convolution
    x = Conv2D(512, 3, padding='same')(x)
    x = apply_noise(x)
    x = WeightDemod(512)([x, style_input])
    x = LeakyReLU(0.2)(x)

    # Upsampling blocks with increased complexity, stopping at 64x64
    for filters, size in zip([512, 256, 128, 64], [8, 16, 32, 64]):  # Adjusted to stop at 64x64
        x = UpSampling2D()(x)
        x = Conv2D(filters, 3, padding='same')(x)
        x = apply_noise(x)
        x = WeightDemod(filters)([x, style_input])
        x = LeakyReLU(0.2)(x)

        # Adding an additional convolutional layer for more complexity
        x = Conv2D(filters, 3, padding='same')(x)
        x = apply_noise(x)
        x = LeakyReLU(0.2)(x)

    # Final convolution to get the desired number of channels (NUM_CHANNELS)
    x = Conv2D(NUM_CHANNELS, 1, activation='tanh')(x)

    return Model([constant_input, style_input], x, name="synthesis_network")



def build_generator():
    latent_input = Input(shape=(LATENT_DIM,))
    constant_input = Input(shape=(INITIAL_SIZE, INITIAL_SIZE, LATENT_DIM))

    mapping_network = build_mapping_network()
    synthesis_network = build_synthesis_network()

    w = mapping_network(latent_input)
    output = synthesis_network([constant_input, w])

    return Model([latent_input, constant_input], output, name="generator")

def build_discriminator():
    input_image = Input(shape=(FINAL_SIZE, FINAL_SIZE, NUM_CHANNELS))
    
    x = Conv2D(64, 3, strides=2, padding='same')(input_image)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, 3, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(1)(x)

    return Model(input_image, x, name="discriminator")

def build_stylegan():
    latent_input = Input(shape=(LATENT_DIM,))
    constant_input = Input(shape=(INITIAL_SIZE, INITIAL_SIZE, LATENT_DIM))

    generator = build_generator()
    discriminator = build_discriminator()

    generated_image = generator([latent_input, constant_input])
    validity = discriminator(generated_image)

    return Model([latent_input, constant_input], validity, name="stylegan2")

# Build and summarize the models
generator = build_generator()
discriminator = build_discriminator()
stylegan2 = build_stylegan()

print("Generator Summary:")
generator.summary()

print("\nDiscriminator Summary:")
discriminator.summary()

print("\nStyleGAN2 Summary:")
stylegan2.summary()
