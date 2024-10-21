from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

# List of filters to be applied in the encoding and reverse in the decoding step 
filterList = [64,128,256,512]

# Function that does the encoding of the Unet by applying convolutions and max pooling to the given tensor at each filter level
# Parameters: inputTensor, a tf tensor that will be encoded
# Returns: (tensor, skipConnectionList), a tuple containing the resulting encoded tensor and a list of the skip connections at each step
def encoder(inputTensor):
    skipConnectionList = []
    tensor = inputTensor
    for filter in filterList: 
        firstConv = Conv2D(filter, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(tensor)
        secondConv = Conv2D(filter, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(firstConv)
        skipConnectionList.append(secondConv)
        tensor = MaxPooling2D(pool_size = (2,2), padding = 'same')(secondConv)
    return tensor, skipConnectionList

# Function that does the decoding of the Unet by applying an up convolution followed by concatting the tensor with the skip connection and then applying convolutions.
# Parameters: inputTensor, a tf tensor that will be decoded
# Returns: tensor, a decoded tensor 
def decoder(skipConnectionList, inputTensor):
    tensor = inputTensor
    for filter in reversed(filterList):
        upConv = Conv2DTranspose(filter, kernel_size = (2,2), padding = 'same', activation = 'relu', strides = 2)(tensor)
        skipConnection = skipConnectionList.pop()
        concatTensor = concatenate([upConv, skipConnection])
        firstConv = Conv2D(filter, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(concatTensor)
        secondConv = Conv2D(filter, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(firstConv)
        tensor = secondConv
    finalConv = Conv2D(1, kernel_size=(1, 1), padding='same', strides=1, activation='sigmoid')(tensor)
    return finalConv

# Function that applies the bottleneck of the unet. 
# Parameters: inputTensor, a tf tensor that will be decoded
# Returns: tensor, a tensor that has gone through two convolutions 
def bottleneck(inputTensor):
        firstConv = Conv2D(1024, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(inputTensor)
        secondConv = Conv2D(1024, kernel_size = (3,3), padding = 'same', strides = 1, activation = 'relu')(firstConv)
        tensor = secondConv
        return tensor

# Function that Applies the encoder, bottleneck and decoder into one unet model
# Returns a keras model of the unet
def unet(): 
    inputs = Input(shape = (256, 128, 1))
    encodedResult, skipConnectionList = encoder(inputs)
    bottleneckResult = bottleneck(encodedResult)
    decodedResult = decoder(skipConnectionList, bottleneckResult)
    return Model(inputs=[inputs], outputs=[decodedResult])
