import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K


#define training variables
BATCH_SIZE = 16
EPOCHS = 50
n_classes = 6
learning_rate = 0.0001
run = "drop0.2"


# Define the Dice similarity coefficient
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)  # Flatten ground truth tensor
    y_pred_f = K.flatten(y_pred)  # Flatten predicted tensor
    intersection = K.sum(y_true_f * y_pred_f)  # Intersection between true and predicted
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)