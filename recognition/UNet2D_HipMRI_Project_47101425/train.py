from modules import UNet
from dataset import MedicalImageDataset
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Num GPUs Available: {len(gpus)}")
else:
    print("No GPUs available, using CPU.")


def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    dice = (2. * intersection + epsilon) / (tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) + epsilon)
    return tf.reduce_mean(dice)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
    return bce + d_loss

input_dims = (256, 144, 1) 
model = UNet(input_dims=input_dims)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss=combined_loss, metrics=[dice_coefficient])

image_dir = "C:/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/HipMRI_study_keras_slices_data/keras_slices_train"
train_dataset = MedicalImageDataset(image_dir=image_dir, normImage=True, batch_size=8, shuffle=True)
dataset = train_dataset.get_dataset()

model.fit(dataset, epochs=1, steps_per_epoch=len(dataset))

model.summary()
model.save('unet_model', save_format='tf')
