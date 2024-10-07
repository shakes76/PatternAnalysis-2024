import sys
sys.path.insert(1, './Modules')

import sklearn.model_selection
import random
import numpy as np
import keras
import tensorflow as tf
import sklearn
import pandas as pd

from modules import SiameseNetwork, ContrastiveLoss
from Modules import NeuralNetwork
from dataset import DicomDataset, ImageUniformityOptions

epochs = 20
batch_size = 16
margin = 1  # Margin for contrastive loss.

dataset = DicomDataset("datasets/train", ImageUniformityOptions.RESIZE, limit=256, resize_size=(32, 32), stratify=True)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset.dicom_images, dataset.dicom_labels, test_size=0.3)

# Change the data type to a floating point format
x_train_val = x_train.astype("float32")
x_test = x_test.astype("float32")

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

base_model = NeuralNetwork.FunctionalNetwork()
base_model.add_training_data(x_train_1, None)
base_model.add_input_layer()
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
network.add_training_data(x_train_1, None)

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
network.add_training_data([x_train_1, x_train_2], labels_train)
network.add_testing_data([x_test_1, x_test_2], labels_test)
network.fit(verbose=1)

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