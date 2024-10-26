import sys
sys.path.insert(1, './Modules')

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.manifold import TSNE
import tensorflow_datasets as tfds
import numpy as np
import torch
from pytorch_metric_learning import losses, miners

from dataset import BalancedMelanomaDataset, FullMelanomaDataset

from Modules import NeuralNetwork

# paths for monitoring
WANDB_SIMILARITY_PATH = "melanoma-similarity-mobile"
WANDB_CLASSIFICATION_PATH = "melanoma-classification-mobile"

# hyperparameters
BATCH_SIZE = 256
MARGIN = 0.4
IMAGE_SHAPE = (256, 256)
VALIDATION_SPLIT = 0.1
TESTING_SPLIT = 0.2
BALANCE_SPLIT = 0.5
LAYERS_TO_UNFREEZE = -1
LEARNING_RATE = 0.001
EMBEDDINGS_EPOCHS = 20
CLASSIFICATION_EPOCHS = 80

# similarity_loss = losses.TripletMarginLoss(margin=MARGIN)
# similarity_optim = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# miner = miners.TripletMarginMiner(margin=MARGIN, type_of_triplets="semihard")
similarity_loss = tfa.losses.TripletSemiHardLoss()
similarity_optim = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

classification_loss = tf.keras.losses.BinaryCrossentropy()
classification_optim = tf.keras.optimizers.Adam()


# datasets
df = BalancedMelanomaDataset(
    image_shape=IMAGE_SHAPE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    testing_split=TESTING_SPLIT,
    balance_split=BALANCE_SPLIT
)

df_full = FullMelanomaDataset(    
    image_shape=IMAGE_SHAPE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    testing_split=TESTING_SPLIT,
    balance_split=BALANCE_SPLIT
)

# grab the relevent dataset
dataset = df.dataset
dataset_val = df.dataset_val
dataset_test = df.dataset_test

# grab the pretrained model
pretrained_model = tf.keras.applications.InceptionV3(
    include_top=False,
    input_shape=(*IMAGE_SHAPE, 3)
    )
pretrained_model.trainable = False

# unfreeze the last few layers
for layer in pretrained_model.layers[LAYERS_TO_UNFREEZE:]:
    layer.trainable = True

# embeddings network
base_model = NeuralNetwork.FunctionalNetwork()
base_model.add_generic_layer(tf.keras.layers.Input(shape=(*IMAGE_SHAPE, 3)))
base_model.add_generic_layer(pretrained_model)
base_model.add_dropout(0.3)
base_model.add_global_pooling2D_layer("max")
base_model.add_dense_layer(2048, activation="leaky_relu")
base_model.add_dropout(0.1)
base_model.add_dense_layer(1024, activation="leaky_relu")
base_model.add_dense_layer(1024, activation="leaky_relu")
base_model.add_generic_layer(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x)))

# set hyperparameters of network
base_model.set_loss_function(similarity_loss)
base_model.set_optimisation(similarity_optim)
base_model.set_epochs(EMBEDDINGS_EPOCHS)
base_model.enable_wandb(WANDB_SIMILARITY_PATH)
# base_model.set_early_stopping("val_loss", patience=20, min_delta=0.01)

# generate network
base_model.generate_functional_model()
base_model.compile_functional_model()

# # Custom training loop
# def train_step(images, labels, model, optimizer):
#     with tf.GradientTape() as tape:
#         # Generate embeddings with the TensorFlow model
#         embeddings = base_model.model(images, training=True)
        
#         # Convert embeddings to PyTorch for mining
#         embeddings_pt = torch.tensor(embeddings.numpy(), requires_grad=True)
#         labels_pt = torch.tensor(labels.numpy().ravel())
        
#         # Use PyTorch miner to select triplets
#         hard_triplets = miner(embeddings_pt, labels_pt)
        
#         # Calculate the loss using the selected triplets
#         loss = similarity_loss(embeddings_pt, labels_pt, hard_triplets)
        
#         # Backpropagate the loss into TensorFlow
#         loss_tf = tf.convert_to_tensor(loss.item())
#         grads = tape.gradient(loss_tf, model.trainable_variables)
        
#         # Apply gradients to update weights
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
#     return loss_tf

# for images, labels in dataset:
#     loss = train_step(images, labels, base_model.model, similarity_optim)
#     print("Loss:", loss.numpy())

# fit the network
base_model.fit_model_batches(
    dataset,
    dataset_val,
    verbose=1, 
    class_weight={0: 1.0, 1: 1.0}
    )

# plot TSNE
outputs = base_model.model.predict(dataset_val)
classes = []

for batch in dataset_val:
    features, labels = batch
    numpy_labels = labels.numpy().ravel()
    classes += list(numpy_labels)

classes = np.array(classes)
tsne = TSNE(n_components=2)
embedded = tsne.fit_transform(outputs)
plt.scatter(embedded[:, 0], embedded[:, 1], c=classes)
plt.savefig("tsne.png")

# freeze embeddings
base_model.model.trainable = False

# classifier network
classifier_model = NeuralNetwork.NeuralNetwork()
classifier_model.add_generic_layer(base_model.model)
classifier_model.add_dense_layer(32)
classifier_model.add_dense_layer(16)
classifier_model.add_dense_layer(1, activation="sigmoid")

# set classifier hyperparameters
classifier_model.set_loss_function(classification_loss)
classifier_model.set_epochs(CLASSIFICATION_EPOCHS)
classifier_model.set_batch_size(BATCH_SIZE)
classifier_model.set_optimisation(classification_optim)
# classifier_model.set_early_stopping("val_loss", patience=20, min_delta=0.01)

# add metrics to monitor
classifier_model.add_metric(["accuracy"])
classifier_model.add_metric(tf.keras.metrics.Precision())
classifier_model.add_metric(tf.keras.metrics.Recall())
classifier_model.add_metric(tf.keras.metrics.TruePositives())
classifier_model.add_metric(tf.keras.metrics.TrueNegatives())
classifier_model.add_metric(tf.keras.metrics.FalsePositives())
classifier_model.add_metric(tf.keras.metrics.FalseNegatives())

# enable logging
classifier_model.enable_tensorboard()
classifier_model.enable_wandb()
classifier_model.enable_model_checkpoints("./checkpoints", save_best_only=True)

# compile and run model
classifier_model.compile_model()
classifier_model.fit_model_batches(
    dataset, 
    dataset_val, 
    verbose=1
    )

# evaluate model
print("--- Testing Performance ---")
classifier_model.model.evaluate(dataset_test)

# plot graphs
classifier_model.visualise_training(to_file=True, filename="classification.png")
plt.show()