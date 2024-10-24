import sys
sys.path.insert(1, './Modules')

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

from dataset import BalancedMelanomaDataset, FullMelanomaDataset

from Modules import NeuralNetwork

batch_size = 64
margin = 1  # Margin for contrastive loss.
image_shape = (128, 128)

df = BalancedMelanomaDataset(
    image_shape=image_shape,
    batch_size=64,
    validation_split=0.2
)

df_full = FullMelanomaDataset(    
    image_shape=image_shape,
    batch_size=batch_size,
    validation_split=0.001,
    balance_split=0.5
)

dataset = df_full.dataset
dataset_val = df_full.dataset_val

pretrained_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(*image_shape, 3)
    )
pretrained_model.trainable = False

for layer in pretrained_model.layers[:-1]:
    layer.trainable = True

inputs = tf.keras.layers.Input(shape=(*image_shape, 3))
pretrained_output = pretrained_model(inputs)
finetune_output = tf.keras.layers.Dense(512, activation='leaky_relu')(tf.keras.layers.GlobalMaxPool2D()(pretrained_output))
finetune2_output = tf.keras.layers.Dropout(0.4)(finetune_output)
finetune3_output = tf.keras.layers.Dense(256, activation='leaky_relu')(finetune2_output)
finetune4_output = tf.keras.layers.Dense(128, activation='leaky_relu')(finetune3_output)
normalised_output = finetune4_output

model = tf.keras.Model(inputs, normalised_output)

base_model = NeuralNetwork.FunctionalNetwork()
base_model.model = model

base_model.set_loss_function(tfa.losses.TripletSemiHardLoss(margin=margin))
base_model.set_optimisation(tf.keras.optimizers.Adam(learning_rate = 0.001))
base_model.compile_functional_model()
base_model.summary()

base_model.set_epochs(300)
base_model.set_early_stopping("val_loss", patience=20, min_delta=0.01)
base_model.fit_model_batches(dataset, dataset_val, verbose=1)
base_model.visualise_training(to_file=True, filename="similarity.png")
plt.show()

base_model.model.trainable = False
classifier_model = NeuralNetwork.NeuralNetwork()
classifier_model.add_generic_layer(base_model.model)
classifier_model.add_dense_layer(32)
classifier_model.add_dense_layer(16)
classifier_model.add_dense_layer(1, activation="sigmoid")
classifier_model.set_loss_function(tf.keras.losses.BinaryCrossentropy())
classifier_model.set_epochs(300)
classifier_model.set_early_stopping("val_loss", patience=20, min_delta=0.01)
classifier_model.set_batch_size(128)
classifier_model.set_optimisation("adam")
classifier_model.add_metric(["accuracy"])
classifier_model.add_metric(tf.keras.metrics.Precision())
classifier_model.add_metric(tf.keras.metrics.Recall())

classifier_model.enable_tensorboard()
classifier_model.enable_wandb("balanced-melanoma-")
classifier_model.enable_model_checkpoints("./checkpoints", save_best_only=True)

dataset = dataset.prefetch(tf.data.AUTOTUNE)
dataset_val = dataset.prefetch(tf.data.AUTOTUNE)

classifier_model.compile_model()

classifier_model.fit_model_batches(dataset, dataset_val, verbose=1)
print("--- Testing Performance ---")
classifier_model.model.evaluate(dataset_val)

classifier_model.visualise_training(to_file=True, filename="classification.png")
plt.show()