from modules import build_model
from dataset import get_test_dataset, get_train_validation_dataset, extract_from_dataset
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

tf.debugging.set_log_device_placement(True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("Loading training data and test data...")
train_dataset, val_dataset = get_train_validation_dataset()
test_dataset = get_test_dataset()
train_images, train_labels = extract_from_dataset(train_dataset)
val_images, val_labels = extract_from_dataset(val_dataset)
test_images, test_labels = extract_from_dataset(test_dataset)
print(f"Total number of training labels: {len(train_labels)}")

print("Converting labels to integers...")
encoder = LabelEncoder()
# 只有第一次处理数据的时候使用fit
train_labels = encoder.fit_transform(train_labels)
test_labels = encoder.transform(test_labels)

print("Converting labels to one-hot encoding...")
# encoder.OneHotEncoder()
# train_labels = encoder.fit_transform(train_labels)
# test_labels = encoder.transform(test_labels)
train_labels = tf.keras.utils.to_categorical(train_labels, 2)
test_labels = tf.keras.utils.to_categorical(test_labels, 2)
print(f"train_labels shape: {train_labels.shape}")
print(f"test_labels shape: {test_labels.shape}")

print("Building model...")
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callback function
print("Setting up callbacks...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    tf.keras.callbacks.ModelCheckpoint('best_mode.keras', save_best_only=True, monitor='val_accuracy')
]

# Training model
print("Start model training...")
with tf.device('/GPU:0'):
    history = model.fit(
        train_images, train_labels,
        epochs=50,
        batch_size=16,
        validation_split=0.2, #around 0.1-0.3
        callbacks=callbacks,
        verbose=1
    )
print("Model training completed.")

print("Rvaluating model on test data...")
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

# Save the trained model
model.save('final_model.keras')
print("Model saved as final_model.keras")

# Plot loss curve
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot accuracy curve
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()