from Modules import DataManager, GenericModel, NeuralNetwork
import tensorflow as tf
import numpy as np
from keras import ops

K = tf.keras.backend

class ContrastiveLoss():
    def __new__(cls, margin):
        def contrastive_loss(ytrue, ypred):
            square_pred = ops.square(ypred)
            margin_square = ops.square(ops.maximum(margin - (ypred), 0))
            return ops.mean((1 - ytrue) * square_pred + (ytrue) * margin_square)
        return contrastive_loss

class TripletLoss():
    def __new__(cls, margin):
        def triplet_loss(ytrue, ypred):
            return 0
        return triplet_loss

class SiameseNetwork(NeuralNetwork.FunctionalNetwork):
    def set_basemodel(self, model: NeuralNetwork.FunctionalNetwork):
        self.basemodel = model.get_model()

    def set_contrastivemodel(self, model: NeuralNetwork.FunctionalNetwork):
        self.contrastivemodel = model.get_model()

    def generate_functional_model(self):
        def euclid_dis(vects):
            x,y = vects
            sum_square = K.sum(K.square(x-y), axis=1, keepdims=True)
            return K.sqrt(K.maximum(sum_square, 0))

        def eucl_dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)

        input1 = tf.keras.layers.Input(shape=self.x_train.shape[1:])
        input2 = tf.keras.layers.Input(shape=self.x_train.shape[1:])
        output1 = self.basemodel(input1)
        output2 = self.basemodel(input2)
        distance_layer = tf.keras.layers.Lambda(euclid_dis, output_shape=eucl_dist_output_shape)([output1, output2])
        normal_layer = tf.keras.layers.BatchNormalization()(distance_layer)
        output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(normal_layer)
        self.model = tf.keras.Model([input1, input2], output_layer)