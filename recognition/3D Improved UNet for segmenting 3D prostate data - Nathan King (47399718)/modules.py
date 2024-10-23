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

def decreasing_layer(inputs, stride, filters):
    """
    The decreasing layer represents a level of the UNet that is decreasing.
    Reference [https://arxiv.org/pdf/1802.10508v1]
    input: inputs - model input or previous decreasing layer output
           stride - stride size
           filters - number of filters to output
    output: addition - element-wise addition of convolution and context
    """
    
    #3 x 3 x 3 Convolution
    convolution = layers.Conv3D(filters, (3, 3, 3), strides=stride, padding="same")(inputs)
    
    #Leaky ReLU Activation
    convolution = layers.LeakyReLU(negative_slope=0.01)(convolution)

    #Context Module
    context = context_module(convolution, filters)

    #Element-wise Addition
    addition = layers.Add()([convolution, context])

    return addition

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

def upsampling_layer(localisation, addition, filters):
    """
    The upsampling layer is used to upscale, followed by a halving of the feature maps.
    Reference [https://arxiv.org/pdf/1802.10508v1]
    input: localisation - previous localisation
           addition - previous addition
           filters - number of filters to output
    output: concatenation - produced concatenation
    """
    
    #2 x 2 x 2 Upsampling
    upsample = layers.UpSampling3D((2, 2, 2))(localisation)
    
    #3 x 3 x 3 Stride-1 Convolution
    upsample = layers.Conv3D(filters, (3, 3, 3), padding="same")(upsample)
    
    #Leaky ReLU Activation
    upsample = layers.LeakyReLU(negative_slope=0.01)(upsample)
    
    #Concatenation of lower level and context
    concatenation = layers.Concatenate()([upsample, addition])
    
    return concatenation

def segmentation_layer(localisation, upscale, lower_segmentation, add, filters):
    """
    The segmentation layer segments the previous localisation module output.
    Reference [https://arxiv.org/pdf/1802.10508v1]
    input: localisation - previous localisation
           upscale - whether to upscale
           lower_segmentation - previous segmentation layer
           add: whether to add to lower_segmentation
           filters - number of filters to output
    output: segmentation - produced segmentation
    """
    
    #1 x 1 x 1 Stride-1 Convolution
    segmentation = layers.Conv3D(filters, (1, 1, 1), padding="same")(localisation)
    
    #Leaky ReLU Activation
    #segmentation = layers.LeakyReLU(negative_slope=0.01)(segmentation)
    
    if add == True:
    
        #Element-wise Addition
        segmentation = layers.Add()([lower_segmentation, segmentation])
    
    if upscale == True:
        
        #2 x 2 x 2 Upsampling 
        segmentation = layers.UpSampling3D((2, 2, 2))(segmentation)
    
    return segmentation