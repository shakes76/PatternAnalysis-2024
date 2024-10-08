import sys
sys.path.insert(1, './Modules')

import random
import numpy as np
import keras
from keras import ops
import matplotlib.pyplot as plt
import tensorflow as tf

from modules import SiameseNetwork, ContrastiveLoss
from Modules import NeuralNetwork

epochs = 20
batch_size = 16
margin = 1  # Margin for contrastive loss.


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Change the data type to a floating point format
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train = x_train.reshape(*x_train.shape, 1)
x_test = x_test.reshape(*x_test.shape, 1)

def make_pairs(x, y):
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

    return np.array(pairs), np.array(labels).astype("float32")

# make train pairs
pairs_train, labels_train = make_pairs(x_train, y_train)

# make test pairs
pairs_test, labels_test = make_pairs(x_test, y_test)

x_train_1 = pairs_train[:, 0]  # x_train_1.shape is (60000, 28, 28)
x_train_2 = pairs_train[:, 1]

x_test_1 = pairs_test[:, 0]  # x_test_1.shape = (20000, 28, 28)
x_test_2 = pairs_test[:, 1]

train_dataset = tf.data.Dataset.from_tensor_slices(((x_train_1, x_train_2), labels_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(((x_test_1, x_test_2), labels_test)).batch(batch_size)

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

contrastive_model = NeuralNetwork.FunctionalNetwork()
contrastive_model.add_training_data(np.ones((1, 10, 1)), None)
contrastive_model.add_input_layer()
contrastive_model.add_batch_norm()
contrastive_model.add_dense_layer(1, activation="sigmoid")
contrastive_model.generate_functional_model()

network = SiameseNetwork()
network.set_input_shape((28, 28, 1))

network.set_basemodel(base_model)
network.set_contrastivemodel(contrastive_model)
network.generate_functional_model()

network.set_loss_function(ContrastiveLoss(margin=margin))
network.add_metric(["accuracy"])
network.set_optimisation("RMSprop")
network.compile_functional_model()
network.summary()

network.enable_tensorboard()
network.enable_wandb("mnist-siamese")
network.enable_model_checkpoints("./checkpoints", save_best_only=True)

network.set_batch_size(batch_size)
network.set_epochs(epochs)
network.set_validation_split_size(0.3)

network.model.fit(
    train_dataset, 
    validation_data=test_dataset,
    # batch_size=batch_size,
    verbose=1,
    epochs=20
)

network.visualise_training()

base_model.model.trainable = False
classifier_model = NeuralNetwork.NeuralNetwork()
classifier_model.add_training_data(x_train.reshape(*x_train.shape, 1), keras.utils.to_categorical(y_train, num_classes=10))
classifier_model.add_testing_data(x_test.reshape(*x_test.shape, 1), keras.utils.to_categorical(y_test, num_classes=10))
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