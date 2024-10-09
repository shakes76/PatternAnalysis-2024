import sys
sys.path.insert(1, './Modules')

import random
import numpy as np
import keras
from keras import ops
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd

from modules import SiameseNetwork, ContrastiveLoss
from Modules import NeuralNetwork

epochs = 50
batch_size = 16
image_shape = (28, 28)
margin = 1  # Margin for contrastive loss.

# labels = pd.read_csv("datasets/train_labels.csv")
# dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     "datasets/images", 
#     labels=list(labels.sort_values("image_name")["target"]), 
#     label_mode="binary",
#     shuffle=True,
#     validation_split=0.2,
#     subset="training",
#     seed=0,
#     image_size=(28, 28),
#     batch_size=None
#     )
# dataset_val = tf.keras.preprocessing.image_dataset_from_directory(
#     "datasets/images", 
#     labels=list(labels.sort_values("image_name")["target"]), 
#     label_mode="binary",
#     shuffle=True,
#     validation_split=0.2,
#     subset="validation",
#     seed=0,
#     image_size=(28, 28),
#     batch_size=None
#     )

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/balanced", 
    labels="inferred", 
    label_mode="binary",
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=0,
    image_size=image_shape,
    batch_size=None
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
    batch_size=None
    )

def pairify_dataset(df):
    x = np.asarray(list(map(lambda x: x[0], tfds.as_numpy(df))))
    y = np.asarray(list(map(lambda x: x[1], tfds.as_numpy(df)))).flatten().astype("int32")

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

    return np.array(pairs)[:, 0], np.array(pairs)[:, 1], np.array(labels).astype("float32")

class1, class2, similarity_label = pairify_dataset(dataset)
class1_val, class2_val, similarity_label_val = pairify_dataset(dataset_val)

dataset_paired = tf.data.Dataset.from_tensor_slices(((class1, class2), similarity_label)).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
dataset_paired_val = tf.data.Dataset.from_tensor_slices(((class1_val, class2_val), similarity_label_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

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

contrastive_model = NeuralNetwork.FunctionalNetwork()
contrastive_model.add_training_data(np.ones((1, 10, 1)), None)
contrastive_model.add_input_layer()
contrastive_model.add_batch_norm()
contrastive_model.add_dense_layer(1, activation="sigmoid")
contrastive_model.generate_functional_model()

network = SiameseNetwork()
network.set_input_shape((*image_shape, 3))

network.set_basemodel(base_model)
network.set_contrastivemodel(contrastive_model)
network.generate_functional_model()

network.set_loss_function(ContrastiveLoss(margin=margin))
network.add_metric(["accuracy"])
network.set_optimisation("RMSprop")
network.compile_functional_model()
network.summary()

network.enable_tensorboard()
network.enable_wandb("mnist-melanoma-balanced")
network.enable_model_checkpoints("./checkpoints", save_best_only=True)

network.set_epochs(epochs)
network.fit_model_batches(dataset_paired, dataset_paired_val, verbose=1)
network.visualise_training()

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
classifier_model.enable_tensorboard("./tensorboard-classi.keras")
classifier_model.enable_model_checkpoints("./checkpoints-classi", save_best_only=True)
classifier_model.enable_wandb("mnist-siamese-classi")

classifier_model.compile_model()
classifier_model.fit(verbose=1)