import numpy as np
from tensorflow import keras
from keras import backend as K

def dsc(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return 2 * intersection / union

def dsc_loss(y_true, y_pred):
    return 1 - dsc(y_true, y_pred)