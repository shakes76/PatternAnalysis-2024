import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from modules import unet_model  
from dataset import load_data_2D  

#Set random seed for reproducability of results
np.random.seed(42)
tf.random.set_seed(42)

#Define dataset paths
data_dir = r'//home//groups//comp3710//HipMRI_Study_open//keras_slices_data'
train_images = os.path.join(data_dir, 'keras_slices_train')
train_labels = os.path.join(data_dir, 'keras_slices_seg_train')

validate_images = os.path.join(data_dir, 'keras_slices_validate')
validate_labels = os.path.join(data_dir, 'keras_slices_seg_validate')

test_images = os.path.join(data_dir, 'keras_slices_test')
test_labels = os.path.join(data_dir, 'keras_slices_seg_test')

#Paths to save preprocessed data
X_train_file = 'X_train.npy'
y_train_file = 'y_train.npy'
X_val_file = 'X_val.npy'
y_val_file = 'y_val.npy'
X_test_file = 'X_test.npy'
y_test_file = 'y_test.npy'

#Function loads data, either from .npy files or by processing raw data
def load_or_process_data():
    if os.path.exists(X_train_file) and os.path.exists(y_train_file):
        print("Loading data from cached files...")
        X_train = np.load(X_train_file)
        y_train = np.load(y_train_file)
        X_val = np.load(X_val_file)
        y_val = np.load(y_val_file)
        X_test = np.load(X_test_file)
        y_test = np.load(y_test_file)
    else:
        print("Processing and saving data...")
        #Load the data using the load_data_2D function
        X_train = load_data_2D([os.path.join(train_images, f) for f in os.listdir(train_images)], normImage=True, getAffines=False)
        y_train = load_data_2D([os.path.join(train_labels, f) for f in os.listdir(train_labels)], normImage=False, getAffines=False)

        X_val = load_data_2D([os.path.join(validate_images, f) for f in os.listdir(validate_images)], normImage=True, getAffines=False)
        y_val = load_data_2D([os.path.join(validate_labels, f) for f in os.listdir(validate_labels)], normImage=False, getAffines=False)

        X_test = load_data_2D([os.path.join(test_images, f) for f in os.listdir(test_images)], normImage=True, getAffines=False)
        y_test = load_data_2D([os.path.join(test_labels, f) for f in os.listdir(test_labels)], normImage=False, getAffines=False)

        #Save data to .npy files for faster loading next time
        np.save(X_train_file, X_train)
        np.save(y_train_file, y_train)
        np.save(X_val_file, X_val)
        np.save(y_val_file, y_val)
        np.save(X_test_file, X_test)
        np.save(y_test_file, y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = load_or_process_data()

#Reshape data to add channel dimension (e.g., 256x256 with 1 channel for grayscale)
X_train = np.expand_dims(X_train, axis=-1)
y_train = np.expand_dims(y_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
y_val = np.expand_dims(y_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
y_test = np.expand_dims(y_test, axis=-1)

#Normalise data
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0
y_train = y_train / 3
y_val = y_val / 3
y_test = y_test / 3
#Set to 1 if value > 0.5 (white), and to 0 if value <= 0.5 (black)
y_train = np.where(y_train > 0.5, 1, 0)
y_val = np.where(y_val > 0.5, 1, 0)
y_test = np.where(y_test > 0.5, 1, 0)

#Now for data augmentation to prevent overfitting
#Data augmentation settings
data_gen_args = dict(rotation_range=15,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     fill_mode='nearest')

#Create the ImageDataGenerator for the images (X_train)
image_datagen = ImageDataGenerator(**data_gen_args)

#Create the ImageDataGenerator for the masks (y_train)
mask_datagen = ImageDataGenerator(**data_gen_args)

#Fit the generators to the data (required if you use normalization or other preprocessing)
image_datagen.fit(X_train, augment=True)
mask_datagen.fit(y_train, augment=True)

#Create a generator that yields augmented images and masks
def train_generator(image_datagen, mask_datagen, X_train, y_train, batch_size):
    image_gen = image_datagen.flow(X_train, batch_size=batch_size, seed=1)
    mask_gen = mask_datagen.flow(y_train, batch_size=batch_size, seed=1)
    
    while True:
        X_batch = next(image_gen)
        y_batch = next(mask_gen)
        yield (X_batch, y_batch)

batch_size = 8

#Create the training data generator
train_gen = train_generator(image_datagen, mask_datagen, X_train, y_train, batch_size=batch_size)

#Initialise the U-Net model
model = unet_model(input_size=(256, 128, 1))

#Train the model using the data generator
history = model.fit(
    train_gen,
    steps_per_epoch=len(X_train) // batch_size,
    validation_data=(X_val, y_val),
    epochs=50,
    verbose=1
)

#Plot training and validation accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_validation_accuracy.png')  # Save the accuracy plot
plt.show()

#Plot training and validation loss
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_validation_loss.png')  # Save the loss plot
plt.show()

#Save final model
model.save('unet_final_model.keras')

#Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
