import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

def load_and_preprocess_mnist():
    # Load MNIST dataset
    (x_train, _), (x_test, _) = mnist.load_data()

    # Combine train and test sets
    x_combined = np.concatenate([x_train, x_test], axis=0)

    # Resize images to 64x64
    x_resized = tf.image.resize(x_combined[..., np.newaxis], (64, 64)).numpy()

    # Normalize images to [-1, 1] range
    x_normalized = (x_resized.astype(np.float32) - 127.5) / 127.5

    return x_normalized

def create_mnist_dataset(batch_size):
    # Load and preprocess MNIST data
    mnist_data = load_and_preprocess_mnist()

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(mnist_data)
    
    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=60000).batch(batch_size, drop_remainder=True)

    return dataset

def generate_random_inputs(batch_size, latent_dim, initial_size):
    # Generate random latent vectors
    latent_vectors = tf.random.normal((batch_size, latent_dim))
    
    # Generate random constant inputs
    constant_inputs = tf.random.normal((batch_size, initial_size, initial_size, latent_dim))
    
    return latent_vectors, constant_inputs

# Example usage
if __name__ == "__main__":
    BATCH_SIZE = 32
    LATENT_DIM = 512
    INITIAL_SIZE = 4

    # Create MNIST dataset
    mnist_dataset = create_mnist_dataset(BATCH_SIZE)

    # Generate random inputs
    latent_vectors, constant_inputs = generate_random_inputs(BATCH_SIZE, LATENT_DIM, INITIAL_SIZE)

    print("MNIST dataset shape:", next(iter(mnist_dataset)).shape)
    print("Latent vectors shape:", latent_vectors.shape)
    print("Constant inputs shape:", constant_inputs.shape)