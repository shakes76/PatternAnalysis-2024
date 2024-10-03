import tensorflow as tf
import matplotlib.pyplot as plt
from modules import build_generator, build_discriminator, build_stylegan, LATENT_DIM, INITIAL_SIZE, FINAL_SIZE
from dataset import create_adni_dataset, generate_random_inputs
from sklearn.manifold import TSNE
import numpy as np

# Define training parameters
BATCH_SIZE = 32
EPOCHS = 30
TARGET_SIZE = (64, 64)

# Load the models
generator = build_generator()
discriminator = build_discriminator()
stylegan = build_stylegan()

# Load the ADNI dataset
adni_dataset = create_adni_dataset(BATCH_SIZE, TARGET_SIZE)

# Compile the models
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)

# Define the loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy()

@tf.function
def train_step(real_images):
    latent_vectors, constant_inputs = generate_random_inputs(BATCH_SIZE, LATENT_DIM, INITIAL_SIZE)

    # Train generator
    with tf.GradientTape() as gen_tape:
        generated_images = generator([latent_vectors, constant_inputs], training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

    # Train discriminator
    with tf.GradientTape() as disc_tape:
        generated_images = generator([latent_vectors, constant_inputs], training=True)
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = real_loss + fake_loss

    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(epochs):
    for epoch in range(epochs):
        for batch in adni_dataset:
            gen_loss, disc_loss = train_step(batch)
        
        print(f"Epoch {epoch+1}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")

        if (epoch + 1) % 2 == 0:
            generate_and_save_images(generator, epoch + 1)

def generate_and_save_images(model, epoch):
    latent_vectors, constant_inputs = generate_random_inputs(16, LATENT_DIM, INITIAL_SIZE)
    predictions = model([latent_vectors, constant_inputs], training=False)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.close()

def extract_features(dataset, model):
    features = []
    labels = []
    for batch in dataset:
        # Get features from the discriminator
        feature_output = model(batch, training=False)
        features.append(feature_output.numpy())
        # Assuming labels are derived from the directory names
        labels.extend(['AD' if 'AD' in str(batch) else 'NC'] * batch.shape[0])
    return np.concatenate(features), np.array(labels)

if __name__ == "__main__":
    train(EPOCHS)

    # Extract features from the discriminator
    real_features, real_labels = extract_features(adni_dataset, discriminator)

    # Generate some images and extract their features
    latent_vector, constant_input = generate_random_inputs(1000, LATENT_DIM, INITIAL_SIZE)
    generated_images = generator.predict([latent_vector, constant_input])
    generated_features = discriminator.predict(generated_images)

    # Combine real and generated features
    all_features = np.concatenate([real_features, generated_features])
    all_labels = np.concatenate([real_labels, ['Generated'] * generated_features.shape[0]])

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(all_features)

    # Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=[0 if label == 'AD' else 1 for label in all_labels], cmap='viridis', alpha=0.5)
    plt.title('t-SNE Visualization of Real and Generated Images')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter, ticks=[0, 1], label='Label')
    plt.legend(['AD', 'NC', 'Generated'])
    plt.savefig('t-SNE_visualization.png')

    # After training, generate a sample image
    latent_vector, constant_input = generate_random_inputs(1, LATENT_DIM, INITIAL_SIZE)
    generated_image = generator.predict([latent_vector, constant_input])
    print(f"Generated image shape: {generated_image.shape}")

    # Test discriminator
    discriminator_output = discriminator.predict(generated_image)
    print(f"Discriminator output: {discriminator_output}")

    # Test full StyleGAN
    stylegan_output = stylegan.predict([latent_vector, constant_input])
    print(f"StyleGAN output: {stylegan_output}")

    # Save a final generated image
    plt.figure(figsize=(1, 1))
    plt.imshow(generated_image[0, :, :, 0] * 0.5 + 0.5, cmap='gray')
    plt.axis('off')
    plt.savefig('final_generated_image.png')
    plt.close()