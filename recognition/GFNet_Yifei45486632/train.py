from modules import build_model
from dataset import get_test_dataset, get_train_validation_dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

# Loading training data and test data
print("Loading training data and test data...")
train_dataset, val_dataset = get_train_validation_dataset()
test_dataset = get_test_dataset()

# Build model
print("Building model...")
model = build_model()
model.compile(# optimizer='adam',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define callback function
print("Setting up callbacks...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10,
                                     monitor='val_accuracy',
                                     mode='max',
                                     restore_best_weights=False),

    tf.keras.callbacks.ModelCheckpoint('best_mode.keras', 
                                     save_best_only=True, 
                                     monitor='val_accuracy')
]

# Training model
print("Start model training...")
with tf.device('/GPU:0'):
    history = model.fit(
        train_dataset,  
        epochs=50,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
print("Model training completed.")

print("Evaluating model on test data...")
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

# Save the trained model
model.save('final_model.keras')
print("Model saved as final_model.keras")

# Plot and save loss curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
plt.close()

# Plot and save accuracy curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_curve.png')
plt.close()

# Training end
print("Training plots have been saved as 'loss_curve.png' and 'accuracy_curve.png'")