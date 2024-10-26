import tensorflow as tf
import random
import pandas as pd

class BalancedMelanomaDataset():
    '''
    A dataset which has been pre-created to have equal proportions of both classes.
    '''
    def __init__(self, 
                 image_shape=(28, 28), # the shape of the image to resize to
                 batch_size=128, # the batch size
                 testing_split=0.1, # the proportion of the dataset reserved for testing
                 validation_split=0.1, # the proportional of the dataset reserved for validation
                 balance_split=None # the positive/negative class split to oversample to, if not None
                 ):
        
        # pick a random seed
        dataset_seed = random.randint(0, 10000)

        # grab the training dataset
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            "datasets/balanced", 
            labels="inferred", 
            label_mode="binary",
            shuffle=True,
            validation_split=testing_split,
            subset="training",
            seed=dataset_seed,
            image_size=image_shape,
            batch_size=None,
            class_names=["negative", "positive"]
        )

        # grab the testing dataset.
        dataset_test = tf.keras.preprocessing.image_dataset_from_directory(
            "datasets/balanced", 
            labels="inferred", 
            label_mode="binary",
            shuffle=True,
            validation_split=testing_split,
            subset="validation",
            seed=dataset_seed,
            image_size=image_shape,
            batch_size=None,
            class_names=["negative", "positive"]
            )
        
        # normalise both datasets
        dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y))
        dataset_test = dataset_test.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y))

        data_augmenter = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.9),
            tf.keras.layers.RandomBrightness(0.1/255),
            tf.keras.layers.RandomZoom(0.3),
            tf.keras.layers.GaussianNoise(0.15)
        ])

        dataset = dataset.map(lambda x, y: (data_augmenter(x, training=True), y))

        num_total = tf.data.Dataset.cardinality(dataset).numpy()
        num_val = int(num_total * validation_split)
        dataset_val = dataset.take(num_val)
        dataset = dataset.skip(num_val).take(num_total - num_val)

        # even out the dataset split
        if balance_split:
            dataset_positive = dataset.filter(lambda x,y: tf.reduce_all(tf.math.equal(y, 1))).repeat().take(1500)
            dataset_negative = dataset.filter(lambda x,y: tf.reduce_all(tf.math.equal(y, 0))).repeat().take(1500)
            
            dataset = dataset_positive.concatenate(dataset_negative).shuffle(3000)
            dataset = dataset.take(3000)

        # batch the data
        dataset = dataset.batch(batch_size)
        dataset_val = dataset_val.batch(batch_size)
        dataset_test = dataset_test.batch(batch_size)

        # prefetch the datasets to increase speed
        self.dataset = dataset.prefetch(tf.data.AUTOTUNE)
        self.dataset_val = dataset_val.prefetch(tf.data.AUTOTUNE)
        self.dataset_test = dataset_test.prefetch(tf.data.AUTOTUNE)

class FullMelanomaDataset():
    '''
    The full dataset, resized down to 256x256
    '''
    def __init__(self, 
                 image_shape=(28, 28), # the image size to scale down to
                 batch_size=128, # the base size
                 testing_split=0.1,
                 validation_split=0.2, # the proportion of the dataset to reserve for validation
                 balance_split=None # the positive/negative class split to oversample to, if not None
                 ):
        
        # pick a random seed
        dataset_seed = random.randint(0, 10000)

        # read in the metadata which tells us what labels there are
        labels = pd.read_csv("datasets/smaller/train-metadata.csv")

        # get the training dataset
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            "datasets/smaller/train-image", 
            labels=list(labels.sort_values("isic_id")["target"]), 
            label_mode="binary",
            shuffle=True,
            validation_split=testing_split,
            subset="training",
            seed=dataset_seed,
            image_size=image_shape,
            batch_size=None
            )
        
        # get the validation dataset
        dataset_test = tf.keras.preprocessing.image_dataset_from_directory(
            "datasets/smaller/train-image", 
            labels=list(labels.sort_values("isic_id")["target"]), 
            label_mode="binary",
            shuffle=True,
            validation_split=testing_split,
            subset="validation",
            seed=dataset_seed,
            image_size=image_shape,
            batch_size=None
            )
        
        # normalise both datasets
        dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y))
        dataset_test = dataset_test.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y))

        data_augmenter = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.9),
            tf.keras.layers.RandomBrightness(0.1/255),
            tf.keras.layers.RandomZoom(0.3),
            tf.keras.layers.GaussianNoise(0.15)
        ])

        num_total = tf.data.Dataset.cardinality(dataset).numpy()
        num_val = int(num_total * validation_split)
        dataset_val = dataset.take(num_val)
        dataset = dataset.skip(num_val).take(num_total - num_val)

        # even out the dataset split
        if balance_split:
            dataset_positive = dataset.filter(lambda x,y: tf.reduce_all(tf.math.equal(y, 1))).repeat().take(1500)
            dataset_negative = dataset.filter(lambda x,y: tf.reduce_all(tf.math.equal(y, 0))).repeat().take(1500)
            
            # dataset = dataset_positive.concatenate(dataset_negative)
            dataset = tf.data.Dataset.sample_from_datasets(
                [dataset_positive, dataset_negative],
                weights=[0.5, 0.5]
            )

            dataset = dataset.take(3000)

        dataset = dataset.map(lambda x, y: (data_augmenter(x, training=True), y))

        # batch the data
        dataset = dataset.batch(batch_size)
        dataset_val = dataset_val.batch(batch_size)
        dataset_test = dataset_test.batch(batch_size)

        # prefetch the datasets to increase speed
        self.dataset = dataset.prefetch(tf.data.AUTOTUNE)
        self.dataset_val = dataset_val.prefetch(tf.data.AUTOTUNE)
        self.dataset_test = dataset_test.prefetch(tf.data.AUTOTUNE)
