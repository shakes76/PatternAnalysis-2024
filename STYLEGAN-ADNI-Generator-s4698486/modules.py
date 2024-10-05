# modules.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, BatchNormalization, Dropout, Flatten

# Define the discriminator model
def define_discriminator(input_shape=(256, 256, 1)):
    model = tf.keras.Sequential()
    
    model.add(Conv2D(32, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(1))
    return model

# Define the loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


######################################################################## MODIFYING GAN INTO STYLEGAN. ###########################################################


# To turn into StyleGAN, we must add some elements.

# 1) Add fully connected layer which converts latent space vector into style space.

# This uses a dimension of 512, 8 blocks and a leaky relu with 0.2 alpha - as perscribed in StyleGAN2 paper.

# NOTE: Also don't want any pixel normalisation. This was used in StyleGAN1 but was removed for #TODO: WHY WAS THIS REMOVED?
"""
def mapping_network(z):
    # Now we are actually taking in our latent space.
    x = z
    for _ in range(8):  # Typically uses 8 fully connected layers
        x = Dense(512)(x)
        x = LeakyReLU(0.2)(x)  # Use LeakyReLU instead of ReLU for more stable gradients
    return x
"""

class MappingNetwork(tf.keras.layers.Layer):
    def __init__(self):
        super(MappingNetwork, self).__init__()
        self.dense_layers = [Dense(512) for _ in range(8)]
        self.activations = [LeakyReLU(0.2) for _ in range(8)]

    def call(self, z):
        x = z
        for dense_layer, activation in zip(self.dense_layers, self.activations):
            x = dense_layer(x)
            x = activation(x)
        return x





# 2) Add AdaIN - Adaptive instance normalisation. This prov
class AdaIN(tf.keras.layers.Layer):
    counter = 0
    def __init__(self, epsilon=1e-5):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon

        #self.counter = 0

    def call(self, inputs):

        #print("\n================== ADAIN =================")

        #print("input from AdaIN:", inputs)
        content, style = inputs
        mean, variance = tf.nn.moments(content, [1, 2], keepdims=True)
        content_norm = (content - mean) / tf.sqrt(variance + self.epsilon)

        num_channels = tf.shape(content)[-1]

        batch_size = tf.shape(content)[0]

        # Bad coding practice to try to handle last line of code where we provide
        # only 'f' as our style block so have no shift.
        #if tf.shape(style)[-1] == num_channels:  # Check if style has the same number of channels as content

        #print("AdaIN batch size =", batch_size)
        #print("AdaIN num channels =", num_channels)
        #print("counter =", AdaIN.counter)
        """
        if num_channels == 64 and self.counter % 2 == 1:
            print("INTO WEIRD CONDITION")
            scale = tf.reshape(style, [batch_size, 1, 1, num_channels])
            shift = tf.zeros_like(scale)  # Use zero shifts if not provided
        else:
            scale, shift = tf.split(style, num_or_size_splits=2, axis=-1)
            # Reshape to match the input tensor's shape for broadcasting
            scale = tf.reshape(scale, [batch_size, 1, 1, num_channels])  # [10, 1, 1, 256]
            shift = tf.reshape(shift, [batch_size, 1, 1, num_channels])  # [10, 1, 1, 256]
        """
        scale, shift = tf.split(style, num_or_size_splits=2, axis=-1)  # Assuming style has shape [batch_size, 512]

        

        # Reshape to match the input tensor's shape for broadcasting
        scale = tf.reshape(scale, [batch_size, 1, 1, num_channels])  # [10, 1, 1, 256]
        shift = tf.reshape(shift, [batch_size, 1, 1, num_channels])  # [10, 1, 1, 256]

        if num_channels == 64:
            AdaIN.counter += 1

        return scale * content_norm + shift




class ModulatedConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ModulatedConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.kwargs = kwargs  # Store additional kwargs for later use
        self.conv = tf.keras.layers.Conv2D(self.filters, self.kernel_size, use_bias=False, **self.kwargs)

    def build(self, input_shape):
        # Explicitly call the build method for self.conv to ensure kernel is initialized
        #print("HELLO2")
        self.conv.build(input_shape) # THIS IS NEEDED SEEMINGLY.
        #print("HELLO3")
        super(ModulatedConv2D, self).build(input_shape) # WORKED LIKE THIS BUT THEN ERRORED ON BELOW MATHS.
    
    def call(self, x, style):

        # Make sure we build self.conv before attempting to access its elements such as its kernel.
        if not self.built:
            #print("HELLO1")
            self.build(x.shape)

        #print("\n================= ModulatedCONV2d =================")
        
        #print("Input shape:", x.shape)
        
        # Get the shape of the input tensor
        batch_size = tf.shape(x)[0]
        input_channels = tf.shape(x)[-1]

        #tf.print("batch size = ", batch_size)
        #tf.print("i.p channels = ", input_channels)

        #print("Batch size:", batch_size.numpy())
        #print("Input channels:", input_channels.numpy())


        #print("Style size before reshape = ", style.shape)

        # Modulate the convolution kernel with the style vector
        style = tf.reshape(style, [batch_size, 1, 1, input_channels, 1])  # batch size, height, width, input channels, output channel.

        #print("Style size after reshape = ", style.shape)

        weight = tf.expand_dims(self.conv.kernel, axis=0)  # Shape: [1, kernel_height, kernel_width, input_channels, output_channels]
        weight = tf.tile(weight, [batch_size, 1, 1, 1, 1])  # Shape: [batch_size, kernel_height, kernel_width, input_channels, output_channels]

        # Note that kernel width x height is 4x4, input channels are 512 and output channels are 256.

        #print("weight shape =", weight.shape)

        weight = weight * style


        # Weight demodulation - this is just to scale back down the results after modulation (without changing the essence of modulation)
        # in order to prevent variance explosion.

        # Axis = 1, 2, 3 is the kernel size (widthxheight) and number of channels.
        demod = tf.math.rsqrt(tf.reduce_sum(tf.square(weight), axis=[1, 2, 3]) + 1e-8)

        #print("DEMOD SHAPE =", demod.shape)

        # Reshape demod to have the dimensions compatible with weight multiplication
        demod = tf.reshape(demod, [batch_size, 1, 1, 1, self.filters]) # We need to reshape it to [batch_size, 1, 1, 1, self.filters] to enable element-wise multiplication with the weight tensor, which has the shape [batch_size, kernel_height, kernel_width, input_channels, self.filters].

        weight = weight * demod


        # Apply convolution with modulated weights using the depthwise convolution approach
        batch, height, width, channels = x.shape
        x = tf.reshape(x, [1, batch * height, width, channels])  # Combine batch and height dimensions

        #print("X reshape for depthwise conv2d =", x.shape)


        # We now need to remove the batch size from the weight, since we combined batch and height dimensions in x - losing a dimension.
        # This loss of dimension is required because conv2d expects 4D tensor input.
        weight = tf.reshape(weight, [self.kernel_size[0] * batch, self.kernel_size[1], channels, self.filters]) # Now we have combined batch size into height for both tensors.

        #print("Weight reshape for depthwise conv2d =", weight.shape)

        # No strided convolution here - leave the striding / upsampling to the upsampling layers iwth bilinear interpolation
        # as outlined in the paper.
        x = tf.nn.depthwise_conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')

        #print("X shape after depthwise_conv2d:", x.shape)

        # Final reshape after depthwise convolution
        batch, new_height, new_width, total_filters = x.shape  # where total_filters = input_channels * self.filters

        # Reshape to [batch_size, height, width, input_channels, self.filters]
        x = tf.reshape(x, [batch_size, height, width, input_channels, self.filters])

        # Now reduce along the input_channels axis, assuming some form of aggregation (e.g., summation)
        x = tf.reduce_sum(x, axis=3)  # Reduce along the input channel axis (axis=3)

        #print("final x shape from modulated conv2d =", x.shape)


        #x = tf.reshape(x, [batch, height, width, self.filters])  # Reshape to original dimensions - back to separating batch size and height.

        return x


# The modulatedConv2d is not using strided convolution with 2x2 kernel unlike in non-style GAN implementation.



# Define the StyleGAN-like Generator with original blocks
class StyleGANGenerator(tf.keras.Model):
    def __init__(self, mapping_network: MappingNetwork):
        super(StyleGANGenerator, self).__init__()

        self.mapping_network = mapping_network

        # Paper says that the initial input to the generator should be generated from ~N(0, 1)
        self.constant_input = tf.Variable(tf.random.normal([1, 4, 4, 512]), trainable=True)

        # Initial block - replaced by random input to begin with.
        self.init_block = tf.keras.Sequential([
            Dense(32 * 32 * 256, input_shape=(512,)),  # Note: Input now uses 512 from mapping network
            LeakyReLU(alpha=0.2),
            Reshape((32, 32, 256))
        ])

        # This is the initialiser, so this defines the different layers.
        self.conv_blocks = []
        self.style_blocks = []
        self.noise_injections = []
        self.upsampling_layers = []

        i = 0

        # Updated filters to transition smoothly to 256x256 resolution
        filters = [256, 128, 64, 32, 16, 8]#, 4]
        
        for i, f in enumerate(filters):
            # Add an upsampling layer
            self.upsampling_layers.append(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear'))
            
            # Add a modulated convolution block
            self.conv_blocks.append(ModulatedConv2D(f, (4, 4), padding='same'))
            
            # Add a style block for scale and shift
            self.style_blocks.append(Dense(2 * f))  # Use 2 * f to split into scale and shift
            
            # Add noise injection
            self.noise_injections.append(tf.keras.layers.GaussianNoise(0.1))
        
        # Final Conv2D layer to output to a single channel.
        self.final_conv = Conv2D(1, (3, 3), strides=(1, 1), padding='same', use_bias=False)
    

    """
     What gets called when we use this model. Hence:
     - Performs style mixing to ensure that no subsection of the style space is being overly relied on.
     - Map latent vector to style space
     - Pass this style space vector through every block in the model.
    """
    def call(self, z1, z2=None, mixing_prob=0.9):
        # Map latent vectors z1 and z2 to style vectors w1 and w2
        w1 = self.mapping_network(z1)
        if z2 is not None and tf.random.uniform(()) < mixing_prob:
            w2 = self.mapping_network(z2)
            # Determine a random cutoff point for style mixing
            mixing_cutoff = tf.random.uniform((), minval=1, maxval=len(self.conv_blocks), dtype=tf.int32)
        else:
            w2 = w1
            mixing_cutoff = len(self.conv_blocks)  # No mixing if z2 is None or not chosen



        #print("latent vector 1 shape:", z1.shape)
        batch_size = tf.shape(z1)[0]  # Get the batch size from the input latent vector
        x = tf.tile(self.constant_input, [batch_size, 1, 1, 1])  # Tile the constant input along the batch dimension. Our input space had a placeholder batch size of 1, we now need to change that to 10.


        # Initial constant input
        #x = self.constant_input

        # Apply style blocks with possible mixing
        for i, (upsample_layer, conv_block, style_block, noise_injection) in enumerate(zip(self.upsampling_layers, self.conv_blocks, self.style_blocks, self.noise_injections)):
        #for i, (conv_block, style_block, noise_injection) in enumerate(zip(self.conv_blocks, self.style_blocks, self.noise_injections)):   

            #print("Building layer", i)
            #print("=====================================")

            x = upsample_layer(x)
            
            # Use w1 for layers before cutoff, and w2 for layers after cutoff
            style = style_block(w1 if i < mixing_cutoff else w2)
            x = conv_block(x, style)                    # I think we are erroring here.
            x = LeakyReLU(alpha=0.2)(x)

            # Noise injection
            noise = noise_injection(tf.random.normal(tf.shape(x)))
            x += noise

            # Style modulation using AdaIN
            scale, shift = tf.split(style, num_or_size_splits=2, axis=-1)
            x = AdaIN()([x, (scale, shift)])

        # Final layer to output an image
        x = self.final_conv(x)

        #print("ABSOLUTE FINAL X SHAPE =", x.shape)

        #x = tf.reshape(x, [batch_size, 256, 256, 1])

        return x

# I suppose I could set off a bunch of models doing different training.



# TODO: Summary of StyleGAN1 to StyleGAN2 changes and why.
