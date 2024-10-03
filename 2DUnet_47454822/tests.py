import keras
import numpy as np
from keras.src.optimizers import Adam

from dataset import load_dir, load_data_2D
from tensorflow.keras.preprocessing import image
from modules import unet_model

# [BATCH_SIZE, CHANNELS, HEIGHT, WIDTH]
# load_dir return: [BATCH_SIZE, HEIGHT, WIDTH]

def invoke():
        result = load_data_2D(["/home/ja/Documents/uni/COMP3710/PatternAnalysis-2024/data/keras_slices_seg_train/seg_004_week_0_slice_0.nii.gz",
                               "/home/ja/Documents/uni/COMP3710/PatternAnalysis-2024/data/keras_slices_seg_train/seg_004_week_0_slice_1.nii.gz"])
        print(result)

def invoke_dir():
    X = load_dir("/home/ja/Documents/uni/COMP3710/PatternAnalysis-2024/data/keras_slices_test/")

# def test_experimental():
#
#     train_batches = image.ImageDataGenerator(
#         preprocessing_function=keras.applications.vgg16.preprocess_input,
#         horizontal_flip=True,
#         rescale=1./255
#     ).flow_from_directory(
#         directory="/home/ja/Documents/uni/COMP3710/PatternAnalysis-2024/data",
#         target_size=(224, 224),
#         batch_size=10,
#         shuffle=True,
#     )
#
#     print(train_batches.filenames)

def test_train():
    X = load_dir("/home/ja/Documents/uni/COMP3710/PatternAnalysis-2024/data/keras_slices_test/")
    y = load_dir("/home/ja/Documents/uni/COMP3710/PatternAnalysis-2024/data/keras_slices_seg_test/")

    batch_size = 32

    X = np.expand_dims(X, axis=-1)  # Add a channel dimension, making the shape (batch_size, 256, 128, 1)
    # X = np.resize(X, (batch_size, 128, 128, 1))

    y = np.expand_dims(y, axis=-1)
    # y = np.resize(y, (batch_size, 128, 128, 1))

    print(f"X Shape: {np.shape(X)}")
    print(f"y Shape: {np.shape(y)}")


    model = unet_model((256, 128, 1), batch_size=batch_size)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x=X, y=y, batch_size=batch_size, epochs=1, shuffle=True, verbose=2)

if __name__ == '__main__':
    # invoke_dir()
    test_train()


# POL7010 Dynamics of Governance