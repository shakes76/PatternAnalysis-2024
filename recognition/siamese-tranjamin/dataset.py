import tensorflow as tf
import random
import pandas as pd

class BalancedMelanomaDataset():
    def __init__(self, 
                 image_shape=(28, 28), 
                 batch_size=128,
                 validation_split=0.2,
                 balance_split=None):
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
            batch_size=None,
            class_names=["negative", "positive"]
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
            batch_size=None,
            class_names=["negative", "positive"]
            )

        dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y))
        dataset_val = dataset_val.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y))

        if balance_split is not None:
            dataset_positive = dataset.filter(lambda x,y: tf.reduce_all(tf.math.equal(y, 1)))
            dataset_negative = dataset.filter(lambda x,y: tf.reduce_all(tf.math.equal(y, 0)))

            dataset = tf.data.Dataset.sample_from_datasets(
                [dataset_positive, dataset_negative],
                weights=[balance_split, 1 - balance_split]
            )

            dataset_val = tf.data.Dataset.sample_from_datasets(
                [dataset_positive, dataset_negative],
                weights = [balance_split, 1 - balance_split]
            )

        dataset = dataset.shuffle(10000)
        dataset_val = dataset_val.shuffle(10000)

        dataset = dataset.batch(batch_size)
        dataset_val = dataset_val.batch(batch_size)

        data_augmenter = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.3),
            # tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomZoom(0.4),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.GaussianNoise(0.03)
        ])

        dataset = dataset.map(lambda x, y: (data_augmenter(x, training=True), y))

        self.dataset = dataset.prefetch(tf.data.AUTOTUNE)
        self.dataset_val = dataset_val.prefetch(tf.data.AUTOTUNE)

class FullMelanomaDataset():
    def __init__(self, 
                 image_shape=(28, 28), 
                 batch_size=128,
                 validation_split=0.2,
                 balance_split=None
                 ):
        dataset_seed = random.randint(0, 10000)

        labels = pd.read_csv("datasets/smaller/train-metadata.csv")
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            "datasets/smaller/train-image", 
            labels=list(labels.sort_values("isic_id")["target"]), 
            label_mode="binary",
            shuffle=True,
            validation_split=validation_split,
            subset="training",
            seed=dataset_seed,
            image_size=image_shape,
            batch_size=None
            )
        dataset_val = tf.keras.preprocessing.image_dataset_from_directory(
            "datasets/smaller/train-image", 
            labels=list(labels.sort_values("isic_id")["target"]), 
            label_mode="binary",
            shuffle=True,
            validation_split=validation_split,
            subset="validation",
            seed=dataset_seed,
            image_size=image_shape,
            batch_size=None
            )
        
        dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y))
        dataset_val = dataset_val.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y))

        if balance_split is not None:
            dataset_positive = dataset.filter(lambda x,y: tf.reduce_all(tf.math.equal(y, 1)))
            dataset_negative = dataset.filter(lambda x,y: tf.reduce_all(tf.math.equal(y, 0)))

            dataset = tf.data.Dataset.sample_from_datasets(
                [dataset_positive, dataset_negative],
                weights=[balance_split, 1 - balance_split]
            )

            dataset_val = tf.data.Dataset.sample_from_datasets(
                [dataset_positive, dataset_negative],
                weights = [balance_split, 1 - balance_split]
            )


        dataset = dataset.batch(batch_size)
        dataset_val = dataset_val.batch(batch_size)

        data_augmenter = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.3),
            # tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomZoom(0.4),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.GaussianNoise(0.03)
        ])

        dataset = dataset.map(lambda x, y: (data_augmenter(x, training=True), y))

        self.dataset = dataset.prefetch(tf.data.AUTOTUNE)
        self.dataset_val = dataset_val.prefetch(tf.data.AUTOTUNE)
