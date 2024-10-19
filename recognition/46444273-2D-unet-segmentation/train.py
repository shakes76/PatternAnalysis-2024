from keras.optimizers import Adam
from dataset import get_training_data, get_data_generators
from modules import unet_2d
import matplotlib.pyplot as plt
from util import plot_images_labels, plot_index

model = unet_2d(output_classes=6, input_shape=(256, 128, 1))

opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# fetch training data
train_data, test_data = get_training_data(image_limit=100)

X_train, y_train = train_data
X_test, y_test = test_data

BATCH_SIZE = 4

train_generator, test_generator = get_data_generators(train_data, test_data,
                                                      seed=42, batch_size=BATCH_SIZE)

# check image mask pairs look correct
plot_images_labels(X_train, y_train)
plot_images_labels(X_test, y_test)

# check image-mask pair at index 0
plot_index(0, X_train, y_train)

steps_per_epoch = (len(X_train))//BATCH_SIZE
val_steps_per_epoch = (len(X_test))//BATCH_SIZE

# train the model
history = model.fit(train_generator,
                    validation_data=test_generator,
                    epochs=10, steps_per_epoch=steps_per_epoch,
                    validation_steps=val_steps_per_epoch)

# plot training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot training and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
