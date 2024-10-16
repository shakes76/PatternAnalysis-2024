from modules import build_model
from dataset import load_images
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

print("Loading training data and test data...")
train_images, train_labels = load_images('train')
test_images, test_labels = load_images('test')
print(f"Loaded {len(train_images)} traing images.")
print(f"Shape of the first training image: {train_images[0].shape}")
print(f"Total number of training labels: {len(train_labels)}")

train_images = np.array(train_images)
test_images = np.array(test_images)

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
model.compile(optimizer='adam', loss='categorical_crossentroy',
              metrics=['accuracy'])

# Define callback function
print("Setting up callbacks...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')
]

# Training model
print("Start model training...")
history = model.fit(
    train_images, train_labels,
    epochs=50,
    batch_size=30,
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
model.save('final_model.h5')
print("Model saved as final_model.h5")

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