import sys
sys.path.insert(1, './Modules')

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

from Modules import NeuralNetwork

epochs = 20
batch_size = 64
margin = 1  # Margin for contrastive loss.


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Change the data type to a floating point format
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train = x_train.reshape(*x_train.shape, 1)
x_test = x_test.reshape(*x_test.shape, 1)

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
dataset_val = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

base_model = NeuralNetwork.FunctionalNetwork()
base_model.add_generic_layer(tf.keras.layers.Input(shape=(28, 28, 1)))
base_model.add_batch_norm()
base_model.add_conv2D_layer((5, 5), 4, activation="tanh")
base_model.add_pooling2D_layer("average", (2,2))
base_model.add_conv2D_layer((5, 5), 16, activation="tanh")
base_model.add_pooling2D_layer("average", (2, 2))
base_model.add_flatten_layer()
base_model.add_batch_norm()
base_model.add_dense_layer(10, activation="tanh")
base_model.generate_functional_model()

base_model.set_loss_function(tfa.losses.TripletSemiHardLoss())
def contrastive_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), tf.float32))

base_model.add_metric(contrastive_accuracy)
base_model.set_optimisation("adam")
base_model.compile_functional_model()
base_model.summary()

base_model.set_epochs(epochs)
base_model.fit_model_batches(dataset, dataset_val, verbose=1)
base_model.visualise_training()
plt.show()

base_model.model.trainable = False
classifier_model = NeuralNetwork.NeuralNetwork()
classifier_model.add_training_data(x_train, keras.utils.to_categorical(y_train, num_classes=10))
classifier_model.add_testing_data(x_test, keras.utils.to_categorical(y_test, num_classes=10))
classifier_model.add_generic_layer(base_model.model)
classifier_model.add_dense_layer(32)
classifier_model.add_dense_layer(16)
classifier_model.add_dense_layer(10, activation="softmax")
classifier_model.set_loss_function(tf.keras.losses.CategoricalCrossentropy())
classifier_model.set_epochs(50)
classifier_model.set_batch_size(128)
classifier_model.set_optimisation("adam")
classifier_model.add_metric("accuracy")

classifier_model.enable_tensorboard()
classifier_model.enable_wandb("mnist-siamese-triplet")
classifier_model.enable_model_checkpoints("./checkpoints", save_best_only=True)

classifier_model.compile_model()
classifier_model.fit(verbose=1)
classifier_model.visualise_training()
plt.show()