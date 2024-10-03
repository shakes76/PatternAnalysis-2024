import tensorflow as tf
import numpy as np
import os

def load_and_preprocess_adni(data_dirs, target_size=(64, 64)):
    images = []
    for data_dir in data_dirs:
        for img_name in os.listdir(data_dir):
            img_path = os.path.join(data_dir, img_name)
            img = tf.io.read_file(img_path)
            img = tf.image.decode_png(img, channels=1)  # Assuming PNG images, adjust if needed
            img = tf.image.resize(img, target_size)
            img = tf.cast(img, tf.float32)
            images.append(img)

    # Stack all images into a single tensor
    image_tensor = tf.stack(images)

    # Normalize images to [-1, 1] range
    normalized_images = (image_tensor - 127.5) / 127.5

    return normalized_images

def create_adni_dataset(batch_size, target_size=(64, 64)):
    data_dirs = [
        '/home/groups/comp3710/ADNI/AD_NC/test/AD',
        '/home/groups/comp3710/ADNI/AD_NC/test/NC',
        '/home/groups/comp3710/ADNI/AD_NC/train/AD',
        '/home/groups/comp3710/ADNI/AD_NC/train/NC'
    ]

    # Load and preprocess all images
    all_images = load_and_preprocess_adni(data_dirs, target_size)

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(all_images)
    
    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=len(all_images)).batch(batch_size, drop_remainder=True)

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
    TARGET_SIZE = (64, 64)

    # Create ADNI dataset
    adni_dataset = create_adni_dataset(BATCH_SIZE, TARGET_SIZE)

    # Generate random inputs
    latent_vectors, constant_inputs = generate_random_inputs(BATCH_SIZE, LATENT_DIM, INITIAL_SIZE)

    print("ADNI dataset shape:", next(iter(adni_dataset)).shape)
    print("Latent vectors shape:", latent_vectors.shape)
    print("Constant inputs shape:", constant_inputs.shape)