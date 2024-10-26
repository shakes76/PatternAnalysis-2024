import sys
sys.path.insert(1, './Modules')

from Modules import NeuralNetwork
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.manifold import TSNE
import numpy as np

class SiameseNetwork():
    '''
    A network which has a Siamese architecture.
    '''
    # hyperparameters
    MARGIN = 0.4
    LAYERS_TO_UNFREEZE = -1
    LEARNING_RATE = 0.001
    EMBEDDINGS_EPOCHS = 10
    CLASSIFICATION_EPOCHS = 20

    # loss functions to use for similarity
    similarity_loss = tfa.losses.TripletSemiHardLoss(margin=MARGIN)
    similarity_optim = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # loss functions to use for classification
    classification_loss = tf.keras.losses.BinaryCrossentropy()
    classification_optim = tf.keras.optimizers.Adam()

    def __init__(self,
                 image_shape
                 ):
        '''
        Parameters:
            image_shape (tuple): a tuple of the 2D shape of the input images
        '''
        # define the embeddings model
        self.base_model = NeuralNetwork.FunctionalNetwork()

        # grab the pretrained model
        self.backbone = tf.keras.applications.InceptionV3(
            include_top=False,
            input_shape=(*image_shape, 3)
            )

        # create similarity network
        self.base_model.add_generic_layer(tf.keras.layers.Input(shape=(*image_shape, 3)))
        self.base_model.add_generic_layer(self.backbone)
        self.base_model.add_dropout(0.3)
        self.base_model.add_global_pooling2D_layer("max")
        self.base_model.add_dense_layer(2048, activation="leaky_relu")
        self.base_model.add_dropout(0.2)
        self.base_model.add_dense_layer(2048, activation="leaky_relu")
        self.base_model.add_generic_layer(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x)))

        # generate a functional model
        self.base_model.generate_functional_model()
    
        # set hyperparameters
        self.base_model.set_loss_function(SiameseNetwork.similarity_loss)
        self.base_model.set_optimisation(SiameseNetwork.similarity_optim)
        self.base_model.set_epochs(SiameseNetwork.EMBEDDINGS_EPOCHS)

        # classifier network
        classifier_model = NeuralNetwork.NeuralNetwork()
        classifier_model.add_generic_layer(self.base_model.model)
        classifier_model.add_dense_layer(32)
        classifier_model.add_dense_layer(16)
        classifier_model.add_dense_layer(1, activation="sigmoid")

        # set classifier hyperparameters
        classifier_model.set_loss_function(SiameseNetwork.classification_loss)
        classifier_model.set_epochs(SiameseNetwork.CLASSIFICATION_EPOCHS)
        classifier_model.set_optimisation(SiameseNetwork.classification_optim)

        # add metrics to monitor
        classifier_model.add_metric(["accuracy"])
        classifier_model.add_metric(tf.keras.metrics.Precision())
        classifier_model.add_metric(tf.keras.metrics.Recall())
        classifier_model.add_metric(tf.keras.metrics.TruePositives())
        classifier_model.add_metric(tf.keras.metrics.TrueNegatives())
        classifier_model.add_metric(tf.keras.metrics.FalsePositives())
        classifier_model.add_metric(tf.keras.metrics.FalseNegatives())
        classifier_model.add_metric(tf.keras.metrics.AUC(curve="PR"))

        self.classifier = classifier_model

        # enable mixed precision
        # NeuralNetwork.NeuralNetwork.enable_mixed_precision()
        
    def train_similarity(self, dataset, dataset_val):
        # compile model
        self.base_model.compile_functional_model()

        # unfreeze the last few layers
        self.backbone.trainable = False
        for layer in self.backbone.layers[SiameseNetwork.LAYERS_TO_UNFREEZE:]:
            layer.trainable = True

        self.base_model.fit_model_batches(
            dataset, 
            dataset_val, 
            verbose=1
        )

    def train_classification(self, dataset, dataset_val):
        # compile model
        self.classifier.compile_model()

        # freeze layers
        self.backbone.trainable = False

        # fit
        self.classifier.fit_model_batches(
            dataset, 
            dataset_val, 
            verbose=1
            )
        
    def plot_embeddings(self, data, save_path):
        outputs = self.base_model.model.predict(data)
        classes = []

        for batch in data:
            features, labels = batch
            numpy_labels = labels.numpy().ravel()
            classes += list(numpy_labels)

        classes = np.array(classes)
        tsne = TSNE(n_components=2)
        embedded = tsne.fit_transform(outputs)
        plt.scatter(embedded[:, 0], embedded[:, 1], c=classes)
        plt.savefig(save_path)
    
    def plot_training_classification(self, filename):
        self.classifier.visualise_training(to_file=True, filename=filename)
    
    def plot_training_similarity(self, filename):
        self.classifier.visualise_training(to_file=True, filename=filename)

    def enable_wandb_similarity(self, path):
        self.base_model.enable_wandb(path)
    
    def enable_wandb_classification(self, path):
        self.classifier.enable_wandb(path)
    
    def enable_similarity_checkpoints(self, path, save_best_only=True):
        self.base_model.enable_model_checkpoints(path, save_best_only)

    def enable_classification_checkpoints(self, path, save_best_only=True):
        self.classifier.enable_model_checkpoints(path, save_best_only)

    def predict(self, data):
        self.base_model.model.predict(data)

    def evaluate_classifier(self, data):
        self.classifier.model.evaluate(data)

    def load_classification_model(self, checkpoint):
        self.classifier.compile_model()
        self.classifier.load_checkpoint(checkpoint)