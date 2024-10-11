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
image_shape = (28, 28)

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/balanced", 
    labels="inferred", 
    label_mode="binary",
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=0,
    image_size=image_shape,
    batch_size=batch_size,
    class_names=["positive", "negative"]
)

dataset_val = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/balanced", 
    labels="inferred", 
    label_mode="binary",
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=0,
    image_size=image_shape,
    batch_size=batch_size,
    class_names=["positive", "negative"]
    )

dataset = dataset.shuffle(10000).prefetch(tf.data.AUTOTUNE)
dataset_val = dataset.prefetch(tf.data.AUTOTUNE)

base_model = NeuralNetwork.FunctionalNetwork()
base_model.add_generic_layer(tf.keras.layers.Input(shape=(*image_shape, 3)))
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
base_model.fit_model_batches(dataset, None, verbose=1)
base_model.visualise_training()
plt.show()

base_model.model.trainable = False
classifier_model = NeuralNetwork.NeuralNetwork()
classifier_model.add_generic_layer(base_model.model)
classifier_model.add_dense_layer(32)
classifier_model.add_dense_layer(16)
classifier_model.add_dense_layer(1, activation="sigmoid")
classifier_model.set_loss_function(tf.keras.losses.BinaryCrossentropy())
classifier_model.set_epochs(70)
classifier_model.set_batch_size(128)
classifier_model.set_optimisation("adam")
classifier_model.add_metric(["accuracy"])
classifier_model.add_metric(tf.keras.metrics.Precision())
classifier_model.add_metric(tf.keras.metrics.Recall())

classifier_model.enable_tensorboard()
classifier_model.enable_wandb("balanced-melanoma-")
classifier_model.enable_model_checkpoints("./checkpoints", save_best_only=True)

dataset = dataset.shuffle(10000).prefetch(tf.data.AUTOTUNE)
dataset_val = dataset.prefetch(tf.data.AUTOTUNE)

classifier_model.compile_model()

classifier_model.set_validation_split_size(0.2)
classifier_model.fit_model_batches(dataset, None, verbose=1)
print("--- Testing Performance ---")
classifier_model.model.evaluate(dataset_val)

classifier_model.visualise_training()
plt.show()