import tensorflow as tf
import os

def load_and_preprocess_adni(data_dirs, target_size, batch_size):
    def process_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32)
        img = (img - 127.5) / 127.5
        return img

    # Create a list of image paths
    image_paths = []
    for data_dir in data_dirs:
        for img_name in os.listdir(data_dir):
            img_path = os.path.join(data_dir, img_name)
            image_paths.append(img_path)

    # Check if image_paths is not empty
    if not image_paths:
        raise ValueError("No images found in the specified directories.")

    # Create a TensorFlow dataset from the image paths
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(image_paths)) 
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE) 

    return dataset

def create_adni_dataset(batch_size, target_size):
    # AD class directories
    ad_dirs = [
        '/home/groups/comp3710/ADNI/AD_NC/test/AD',
        '/home/groups/comp3710/ADNI/AD_NC/train/AD'
    ]

    # NC class directories
    nc_dirs = [
        '/home/groups/comp3710/ADNI/AD_NC/test/NC',
        '/home/groups/comp3710/ADNI/AD_NC/train/NC'
    ]

    # Load and preprocess AD and NC images separately
    ad_images = load_and_preprocess_adni(ad_dirs, target_size, batch_size)
    nc_images = load_and_preprocess_adni(nc_dirs, target_size, batch_size)

    return ad_images, nc_images

def generate_random_inputs(batch_size, latent_dim, initial_size):
    # Generate random latent vectors
    latent_vectors = tf.random.normal((batch_size, latent_dim))
    
    # Generate random constant inputs
    constant_inputs = tf.random.normal((batch_size, initial_size, initial_size, latent_dim))
    
    return latent_vectors, constant_inputs



# Example usage
if __name__ == "__main__":
    BATCH_SIZE = 16
    LATENT_DIM = 512
    INITIAL_SIZE = 8
    TARGET_SIZE = (128, 128)

    # Create ADNI dataset for AD and NC classes separately
    ad_images, nc_images = create_adni_dataset(BATCH_SIZE, TARGET_SIZE)

    # Generate random inputs
    latent_vectors, constant_inputs = generate_random_inputs(BATCH_SIZE, LATENT_DIM, INITIAL_SIZE)

    # Check dataset shapes for AD and NC
    print("AD dataset shape:", next(iter(ad_images)).shape)
    print("NC dataset shape:", next(iter(nc_images)).shape)
    print("Latent vectors shape:", latent_vectors.shape)
    print("Constant inputs shape:", constant_inputs.shape)
