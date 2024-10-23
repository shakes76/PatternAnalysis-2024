from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, Input
from keras.models import Model

# pair of convolutional layers with batch normalization applied to each
def convolution_pair(input, filters):
  x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(input)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  return x

# pair of convolution layers, max pooling applied
def encoder_block(input, num_filters):
  x = convolution_pair(input, num_filters)
  p = MaxPooling2D((2, 2))(x)

  return x, p

# upsampled features get concatenated with encoder output (skip features)
def decoder_block(input, skip_features, filters):
  x = Conv2DTranspose(filters, (2, 2), strides=2, padding='same',
                        kernel_initializer='he_normal')(input)
  x = Concatenate()([x, skip_features])
  x = convolution_pair(x, filters)

  return x

# Build UNET model using blocks
def unet_2d(output_classes, input_shape):
  '''
  Construct U-Net architecture, originally described by Ronneberger, Fischer, Brox
  https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

  Params:
    output_classes (int): number of classes to output with softmax
    input_shape (tuple): shape of image data (H x W x C)
  Returns:
    model (keras.models.Model object): U-Net model
  '''
  ins = Input(input_shape)

  # encode step (convolutions kept for skip connections)
  e1, p1 = encoder_block(ins, 64)
  e2, p2 = encoder_block(p1, 128)
  e3, p3 = encoder_block(p2, 256)
  e4, p4 = encoder_block(p3, 512)

  # connect encode and decode pipelines
  bridge = convolution_pair(p4, 1024)

  # decode step (concatenate down convolutions to upsampled convolutions)
  d1 = decoder_block(bridge, e4, 512)
  d2 = decoder_block(d1, e3, 256)
  d3 = decoder_block(d2, e2, 128)
  d4 = decoder_block(d3, e1, 64)

  outs = Conv2D(output_classes, 1, padding='same', activation='softmax')(d4)
  model = Model(ins, outs, name='UNET')

  return model

from keras import backend as K

def dice_similarity(y_true, y_pred):
    y_true_f = K.flatten(y_pred)
    y_pred_f = K.flatten(y_true)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_prostate(y_true, y_pred):
   return dice_similarity(y_true[:,:,:,5], y_pred[:,:,:,5])

def focal_loss(y_true, y_pred):
   
   gamma = 2.0
   alpha = 0.25
   y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
   loss = -y_true * (alpha * K.pow((1 - y_pred), gamma) * K.log(y_pred))
   
   return K.mean(loss)

def f_score(y_true, y_pred, beta=1, smooth=1e-5):

    axes = [1, 2]
    axes.insert(0, 0)

    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp
    fn = K.sum(y_true, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = K.mean(score)

    return score

def dice(y_true, y_pred):
   return 1 - f_score(y_true, y_pred)

def total_loss(y_true, y_pred):
   return dice(y_true, y_pred) + focal_loss(y_true, y_pred)
