import tensorflow as tf
import random

class BalancedMelanomaDataset():
    def __init__(self, 
                 image_shape=(28, 28), 
                 batch_size=128,
                 validation_split=0.2):
        dataset_seed = random.randint(0, 10000)

        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            "datasets/balanced", 
            labels="inferred", 
            label_mode="binary",
            shuffle=True,
            validation_split=validation_split,
            subset="training",
            seed=dataset_seed,
            image_size=image_shape,
            batch_size=batch_size,
            class_names=["positive", "negative"]
        )

        dataset_val = tf.keras.preprocessing.image_dataset_from_directory(
            "datasets/balanced", 
            labels="inferred", 
            label_mode="binary",
            shuffle=True,
            validation_split=validation_split,
            subset="validation",
            seed=dataset_seed,
            image_size=image_shape,
            batch_size=batch_size,
            class_names=["positive", "negative"]
            )

        data_augmenter = tf.keras.Sequential([
            # tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            # tf.keras.layers.RandomRotation(0.2)
        ])

        dataset = dataset.map(lambda x, y: (data_augmenter(x, training=True), y))

        self.dataset = dataset.prefetch(tf.data.AUTOTUNE)
        self.dataset_val = dataset.prefetch(tf.data.AUTOTUNE)

    def training_dataset(self):
        return self.dataset
    
    def validation_dataset(self):
        return self.dataset_val
        