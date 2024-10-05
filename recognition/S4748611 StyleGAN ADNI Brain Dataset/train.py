import tensorflow as tf
import matplotlib.pyplot as plt
from modules import build_generator, build_discriminator, build_stylegan, LATENT_DIM, INITIAL_SIZE, FINAL_SIZE
from dataset import create_adni_dataset, generate_random_inputs
from sklearn.manifold import TSNE
import numpy as np
import wandb

# Define training parameters
BATCH_SIZE = 64
EPOCHS = 10 
TARGET_SIZE = (64, 64)

# Load the models
print("Loading models...")
generator = build_generator()
print("Generator model loaded.")
discriminator = build_discriminator()
print("Discriminator model loaded.")
stylegan = build_stylegan()
print("StyleGAN model loaded.")

# Compile the models
print("Compiling models...")
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)
print("Models compiled.")

# Initialize Weights and Biases
print("Initializing Weights and Biases...")
wandb.init(
    project="StyleGAN ADNI Test", 
    entity="samwolfenden-university-of-queensland",
    config={
        "gen learning rate": 0.0002,
        "disc learning rate": 0.0001,
        "epochs": EPOCHS,
        "optimizer": type(generator_optimizer).__name__,  # Use __name__ to get the name of the class
        "scheduler": type(discriminator_optimizer).__name__,
        "cross entropy loss": type(tf.keras.losses.BinaryCrossentropy()).__name__,
        "name": "SD-ADNI - VAE and Unet",
        "image size": TARGET_SIZE,
        "batch size": BATCH_SIZE
    })

# Load the ADNI dataset
print("Loading ADNI dataset...")
adni_dataset = create_adni_dataset(BATCH_SIZE, TARGET_SIZE)
print("ADNI dataset loaded.")

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
    print("Starting training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}...")
        for batch in adni_dataset:
            gen_loss, disc_loss = train_step(batch)
        
        print(f"Epoch {epoch + 1} completed.")

        # Log losses to Weights and Biases
        wandb.log({"Generator Loss": gen_loss, "Discriminator Loss": disc_loss})
        print("Logged losses to Weights and Biases.")

        if (epoch + 1) % 2 == 0:
            generate_and_save_images(generator, epoch + 1)

    return gen_loss, disc_loss

def generate_and_save_images(model, epoch):
    print(f"Generating and saving images for epoch {epoch}...")
    latent_vectors, constant_inputs = generate_random_inputs(16, LATENT_DIM, INITIAL_SIZE)
    predictions = model([latent_vectors, constant_inputs], training=False)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.close()
    print(f"Images saved for epoch {epoch}.")

def extract_features(dataset, model):
    print("Extracting features from the dataset...")
    features = []
    labels = []
    for batch in dataset:
        # Get features from the discriminator
        feature_output = model(batch, training=False)
        features.append(feature_output.numpy())
        # Assuming labels are derived from the directory names
        labels.extend(['AD' if 'AD' in str(batch) else 'NC'] * batch.shape[0])
    print("Feature extraction completed.")
    return np.concatenate(features), np.array(labels)

def save_models(generator, discriminator, stylegan):
    print("Saving models...")
    generator.save('stylegan_generator.h5')
    discriminator.save('stylegan_discriminator.h5')
    stylegan.save('stylegan_model.h5')
    print("Models saved successfully.")

if __name__ == "__main__":
    gen_loss, disc_loss = train(EPOCHS)

    # Extract features from the discriminator
    real_features, real_labels = extract_features(adni_dataset, discriminator)

    # Generate some images and extract their features
    print("Generating random inputs for image generation...")
    latent_vector, constant_input = generate_random_inputs(1000, LATENT_DIM, INITIAL_SIZE)
    generated_images = generator.predict([latent_vector, constant_input])
    generated_features = discriminator.predict(generated_images)

    # Combine real and generated features
    all_features = np.concatenate([real_features, generated_features])
    all_labels = np.concatenate([real_labels, ['Generated'] * generated_features.shape[0]])

    # Check the shape of all_features
    print("Shape of all_features:", all_features.shape)

    # Apply t-SNE only if there are enough samples and features
    if all_features.shape[0] > 1 and all_features.shape[1] > 1:
        print("Applying t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        embeddings = tsne.fit_transform(all_features)
        print("t-SNE completed.")
    else:
        print("Not enough samples or features for t-SNE.")

    # Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=[0 if label == 'AD' else 1 for label in all_labels], cmap='viridis', alpha=0.5)
    plt.title('t-SNE Visualization of Real and Generated Images')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter, ticks=[0, 1], label='Label')
    plt.legend(['AD', 'NC', 'Generated'])
    plt.savefig('t-SNE_visualization.png')
    print("t-SNE visualization saved.")

    # After training, generate a sample image
    print("Generating a sample image...")
    latent_vector, constant_input = generate_random_inputs(1, LATENT_DIM, INITIAL_SIZE)
    generated_image = generator.predict([latent_vector, constant_input])
    print(f"Generated image shape: {generated_image.shape}")

    # Test discriminator
    print("Testing discriminator on generated image...")
    discriminator_output = discriminator.predict(generated_image)
    print(f"Discriminator output: {discriminator_output}")

    # Test full StyleGAN
    print("Testing full StyleGAN...")
    stylegan_output = stylegan.predict([latent_vector, constant_input])
    print(f"StyleGAN output: {stylegan_output}")

    # Save a final generated image
    plt.figure(figsize=(1, 1))
    plt.imshow(generated_image[0, :, :, 0] * 0.5 + 0.5, cmap='gray')
    plt.axis('off')
    plt.savefig('final_generated_image.png')
    plt.close()
    print("Final generated image saved.")

    # Save the models
    save_models(generator, discriminator, stylegan)

    # Log final outputs to Weights and Biases
    wandb.log({"Generated Image": wandb.Image(generated_image[0])})
    wandb.log({"Discriminator Output": discriminator_output})
    wandb.log({"StyleGAN Output": stylegan_output})
    wandb.log({"Final Generated Image": wandb.Image(generated_image[0])})
    print("Weights and Biases logging completed.")