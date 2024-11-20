import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from dataset import load_data_2D
from train import train_unet
import matplotlib.pyplot as plt


# Function to take a trained unet model and use it for predictions.
def unet_predict(model, data_test):
    """
    Function to predict the model
    Paramaters: 
    - model - the trained unet model to use for the predictions 
    - data_test - the data set to use for predictions
    """
    predictions = model.predict(data_test)
    # Get predicted segmented images
    print(np.argmax(predictions, axis=1))
    print(classification_report(data_test, predictions))

    # Create a plot to plot predictions against tests
    # reference: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    _, ax = plt.subplots(len(predictions), 2)
    for i in range(len(predictions)):
        ax[0, i].set_title(f"[{i}] prediction")
        ax[0, i].im_show(predictions[i].squeeze())
        ax[1, i].set_title("Test Data")
        ax[1, i].im_show(data_test[i].squeeze())
    plt.show()


def main():
    # load and evaluate model reference: https://www.tensorflow.org/tutorials/keras/save_and_load
    # load the trained model
    model = tf.keras.models.load_model('mri_unet.keras')
    model.summary()
    data_train = load_data_2D()
    data_validate = load_data_2D()
    data_test = load_data_2D()
    _, acc = model.evaluate(data_train, data_validate, verbose=2)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
    unet_predict(model, data_test)



if __name__ == "__main__":
    main()