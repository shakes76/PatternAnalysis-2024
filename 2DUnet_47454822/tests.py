import keras
from keras.src.optimizers import Adam

from dataset import load_dir, load_data_2D
from tensorflow.keras.preprocessing import image
from modules import unet_model

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

    model = unet_model((10, 256, 128))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x=X, y=y, batch_size=10, epochs=5, shuffle=True, verbose=2)

if __name__ == '__main__':
    # invoke_dir()
    test_train()