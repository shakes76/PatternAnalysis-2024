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
batch_size = 64
margin = 1  # Margin for contrastive loss.

# labels = pd.read_csv("datasets/train_labels.csv")
# dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     "D:/images2/images", 
#     labels=list(labels.sort_values("image_name")["target"]), 
#     label_mode="binary",
#     shuffle=True,
#     validation_split=0.2,
#     subset="training",
#     seed=0
#     )
# dataset_val = tf.keras.preprocessing.image_dataset_from_directory(
#     "D:/images2/images", 
#     labels=list(labels.sort_values("image_name")["target"]), 
#     label_mode="binary",
#     shuffle=True,
#     validation_split=0.2,
#     subset="validation",
#     seed=0
#     )

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/balanced", 
    labels="inferred", 
    label_mode="binary",
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=0
)

dataset_val = tf.keras.preprocessing.image_dataset_from_directory(
    "datasets/balanced", 
    labels="inferred", 
    label_mode="binary",
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=0
    )

data_augmenter = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2)
])

melanoma_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,      # Randomly rotate images in the range (degrees)
    width_shift_range=0.2,  # Randomly translate images horizontally
    height_shift_range=0.2, # Randomly translate images vertically
    shear_range=0.2,        # Randomly shear images
    zoom_range=0.2,         # Randomly zoom into images
    horizontal_flip=True,    # Randomly flip images
    fill_mode='nearest'     # Fill pixels that are newly created
)

dataset = dataset.map(lambda x, y: (data_augmenter(x, training=True), y))


def make_pairs_batch(x, y):
    num_classes = 2
    class_indices = [tf.where(y == i).numpy() for i in range(num_classes)]

    def get_random_pair(idx1, label1):
        label1 = int(label1)
        
        # matching pair
        idx2 = random.choice(class_indices[label1])
        x2 = x[int(idx2[0])]

        # non-matching pair
        label2 = random.randint(0, 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)
        idx2_nonmatch = random.choice(class_indices[label2])
        x2_nonmatch = x[int(idx2_nonmatch[0])]

        return (x[idx1], x2, x2_nonmatch), (0, 1)

    pairs = []
    labels = []

    for idx1, label1 in enumerate(y.numpy()):  # Convert y to NumPy for iteration
        pos_pair, neg_pair = get_random_pair(idx1, label1)
        pairs.append((pos_pair[0], pos_pair[1]))  # Add matching pair
        pairs.append((neg_pair[0], neg_pair[1]))  # Add non-matching pair
        labels.append(0)
        labels.append(1)

    # Convert the pairs to tensors
    x1 = [pair[0] for pair in pairs]  # Get the first image of each pair
    x2 = [pair[1] for pair in pairs]  # Get the second image of each pair

    # Convert to tensors
    x1_tensor = tf.convert_to_tensor(x1)
    x2_tensor = tf.convert_to_tensor(x2)

    # Ensure shapes are correct (3D)
    if len(x1_tensor.shape) == 3:
        x1_tensor = tf.expand_dims(x1_tensor, axis=0)  # Add batch dimension if missing
    if len(x2_tensor.shape) == 3:
        x2_tensor = tf.expand_dims(x2_tensor, axis=0)  # Add batch dimension if missing

    # Resize to ensure shape is [32, 32, 3]
    x1_resized = tf.image.resize(x1_tensor, size=(256, 256))
    x2_resized = tf.image.resize(x2_tensor, size=(256, 256))

    # Return tensors ensuring the shape is [32, 32, 3] for both x1 and x2
    return x1_resized, x2_resized, tf.convert_to_tensor(labels)

def pair_generation_pipeline(dataset):
    def set_shapes(x1, x2, labels):
        # Set explicit shapes for the tensors
        x1.set_shape([None, 256, 256, 3])  # Image size (32, 32, 3)
        x2.set_shape([None, 256, 256, 3])  # Image size (32, 32, 3)
        labels.set_shape([None])         # Scalar labels (shape (None,))
        return (x1, x2), labels

    return dataset.map(lambda x, y: tf.py_function(
        func=make_pairs_batch, 
        inp=[x, y], 
        Tout=[tf.float32, tf.float32, tf.float32]
    )).map(set_shapes)


# make train pairs
pairs_train = pair_generation_pipeline(dataset)
pairs_val = pair_generation_pipeline(dataset_val)

base_model = NeuralNetwork.FunctionalNetwork()
base_model.add_generic_layer(tf.keras.layers.Input(shape=(256, 256, 3)))
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
contrastive_model.add_generic_layer(tf.keras.layers.Input(shape=(10,)))
contrastive_model.add_batch_norm()
contrastive_model.add_dense_layer(1, activation="sigmoid")
contrastive_model.generate_functional_model()

network = SiameseNetwork()

network.set_basemodel(base_model)
network.set_contrastivemodel(contrastive_model)
network.set_input_shape((256, 256, 3))
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
    pairs_train, 
    validation_data=pairs_val,
    batch_size=batch_size,
    verbose=1,
    epochs=10
    )

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