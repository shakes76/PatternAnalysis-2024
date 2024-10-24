from modules import build_model
from dataset import get_train_validation_dataset, get_test_dataset, extract_labels_from_dataset
import tensorflow as tf
import matplotlib.pyplot as plt

# Load training, validation and test datasets using the custom dataloader
print("Loading training, validation and test datasets...")
train_dataset, val_dataset = get_train_validation_dataset()
test_dataset = get_test_dataset()

train_labels = extract_labels_from_dataset(train_dataset)
test_labels = extract_labels_from_dataset(test_dataset)


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

# Define callback functions
print("Setting up callbacks...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')
]

# Training the model
print("Start model training...")
history = model.fit(
    train_dataset,  # Training dataset from the custom dataloader
    validation_data=val_dataset,  # Validation dataset from the custom dataloader
    epochs=50,
    callbacks=callbacks,
    verbose=1
)
print("Model training completed.")

# Evaluating the model on test dataset
print("Evaluating model on test data...")
test_loss, test_accuracy = model.evaluate(test_dataset)
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
plt.ylabel('Accuracy')
plt.legend()
plt.show()
