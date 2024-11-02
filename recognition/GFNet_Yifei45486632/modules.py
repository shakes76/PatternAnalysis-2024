import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

class GlobalFilterLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalFilterLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialize learnable frequency domain filters
        self.filters = self.add_weight(
            name='freq_filters',
            shape=input_shape[1:],
            initializer='glorot_uniform',
            trainable=True,
            regularizer=tf.keras.regularizers.l2(0.01)
        )

    def call(self, x):
        # Convert to the frequency domain
        x_freq = tf.signal.fft2d(tf.cast(x, tf.complex64))
        
        # Apply a frequency domain filter
        x_filtered = x_freq * tf.cast(self.filters, tf.complex64)
        
        # Convert back to the spatial domain
        x_spatial = tf.signal.ifft2d(x_filtered)
        
        return tf.math.real(x_spatial)

def build_model(input_shape=(224,224,3)):
    # Input layers and data augmentation
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.Sequential([
        layers.RandomRotation(0.2),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ])(inputs)

    # EfficientNetB0 serves as the base feature extractor
    base_model = EfficientNetB0(include_top=False, input_shape=input_shape, pooling='avg')
    x = base_model(x)

    # Add a global frequency filter layer
    x = GlobalFilterLayer()(x)

    # Classification layer
    x = layers.Dense(128, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    predictions = layers.Dense(2, activation='softmax')(x)

    # Model building
    model = models.Model(inputs=inputs, outputs=predictions)
    return model
