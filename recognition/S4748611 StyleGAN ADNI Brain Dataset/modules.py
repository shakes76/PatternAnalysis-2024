import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, Flatten, Layer, UpSampling2D
from tensorflow.keras.models import Model
import numpy as np

# Define constants
LATENT_DIM = 512
INITIAL_SIZE = 4
NUM_CHANNELS = 1
FINAL_SIZE = 64

class AdaIN(Layer):
    def __init__(self, **kwargs):
        super(AdaIN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_channels = input_shape[0][-1]
        self.style_scale_transform = Dense(self.num_channels)
        self.style_shift_transform = Dense(self.num_channels)

    def call(self, inputs):
        x, style = inputs
        style_scale = self.style_scale_transform(style)
        style_shift = self.style_shift_transform(style)
        
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        normalized = (x - mean) / tf.sqrt(variance + 1e-8)
        
        return normalized * (1 + style_scale[:, None, None, :]) + style_shift[:, None, None, :]

def build_mapping_network():
    latent_input = Input(shape=(LATENT_DIM,))
    x = latent_input
    for _ in range(8):
        x = Dense(LATENT_DIM)(x)
        x = LeakyReLU(0.2)(x)
    return Model(latent_input, x, name="mapping_network")

def build_synthesis_network():
    style_input = Input(shape=(LATENT_DIM,))
    constant_input = Input(shape=(INITIAL_SIZE, INITIAL_SIZE, LATENT_DIM))

    x = constant_input
    
    # Initial convolution
    x = Conv2D(512, 3, padding='same')(x)
    x = AdaIN()([x, style_input])
    x = LeakyReLU(0.2)(x)

    # Upsampling blocks
    for i, filters in enumerate([512, 256, 128, 64]):
        x = UpSampling2D()(x)
        x = Conv2D(filters, 3, padding='same')(x)
        x = AdaIN()([x, style_input])
        x = LeakyReLU(0.2)(x)

    # Final convolution to get the desired number of channels
    x = Conv2D(NUM_CHANNELS, 1, activation='tanh')(x)
    
    print(f"Generator output shape: {x.shape}")

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
    
    x = Conv2D(256, 3, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2D(512, 3, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    
    x = Flatten()(x)
    
    x = Dense(512)(x)
    x = LeakyReLU(0.2)(x)
    
    x = Dense(1, activation='sigmoid')(x)

    return Model(input_image, x, name="discriminator")

def build_stylegan():
    latent_input = Input(shape=(LATENT_DIM,))
    constant_input = Input(shape=(INITIAL_SIZE, INITIAL_SIZE, LATENT_DIM))

    generator = build_generator()
    discriminator = build_discriminator()

    generated_image = generator([latent_input, constant_input])
    validity = discriminator(generated_image)

    return Model([latent_input, constant_input], validity, name="stylegan")

# Build and summarize the models
generator = build_generator()
discriminator = build_discriminator()
stylegan = build_stylegan()

print("Generator Summary:")
generator.summary()

print("\nDiscriminator Summary:")
discriminator.summary()

print("\nStyleGAN Summary:")
stylegan.summary()

# Test the models with random inputs
# Test generator
latent_vector = np.random.normal(0, 1, (1, LATENT_DIM))
constant_input = np.random.normal(0, 1, (1, INITIAL_SIZE, INITIAL_SIZE, LATENT_DIM))
generated_image = generator.predict([latent_vector, constant_input])
print(f"\nGenerated image shape: {generated_image.shape}")

# Test discriminator
discriminator_output = discriminator.predict(generated_image)
print(f"Discriminator output: {discriminator_output}")

# Test full StyleGAN
stylegan_output = stylegan.predict([latent_vector, constant_input])
print(f"StyleGAN output: {stylegan_output}")