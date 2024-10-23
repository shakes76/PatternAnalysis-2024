import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def context_module(convolution, filters):
    """
    The context module is defined as a pre-activation residual block with two 3 x 3 x 3 convolutional layers with a dropout layer between.
    Reference [https://arxiv.org/pdf/1802.10508v1]
    input: convolution - previous convolution
           filters - number of filters to output
    output: context - produced context
    """
    
    #Instance Normalisation
    context = layers.GroupNormalization(groups = 1)(convolution)
    
    #3 x 3 x 3 Stride-1 Convolution
    context = layers.Conv3D(filters, (3, 3, 3), padding="same")(context)
    
    #Leaky ReLU Activation
    context = layers.LeakyReLU(negative_slope=0.01)(context)
    
    #Dropout
    context = layers.Dropout(0.3)(context)
    
    #Instance Normalisation
    context = layers.GroupNormalization(groups = 1)(context)
    
    #3 x 3 x 3 Stride-1 Convolution
    context = layers.Conv3D(filters, (3, 3, 3), padding="same")(context)
    
    #Leaky ReLU Activation
    context = layers.LeakyReLU(negative_slope=0.01)(context)
    
    return context

def localisation_module(concatenation, filters):
    """
    The localisation module is defined as a 3 x 3 x 3 convolutional layer followed by a 1 x 1 x 1 convolutional layer.
    Reference [https://arxiv.org/pdf/1802.10508v1]
    input: concatenation - recombined features
           filters - number of filters to output
    output: localisation - produced localisation
    """
    
    #3 x 3 x 3 Stride-1 Convolution
    localisation = layers.Conv3D(filters, (3, 3, 3), padding="same")(concatenation)
    
    #Leaky ReLU Activation
    localisation = layers.LeakyReLU(negative_slope=0.01)(localisation)
    
    #1 x 1 x 1 Stride-1 Convolution
    localisation = layers.Conv3D(filters, (1, 1, 1), padding="same")(localisation)
    
    #Leaky ReLU Activation
    localisation = layers.LeakyReLU(negative_slope=0.01)(localisation)
    
    return localisation