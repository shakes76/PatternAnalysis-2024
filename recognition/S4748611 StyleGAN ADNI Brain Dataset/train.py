import tensorflow as tf
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from modules import build_generator, build_discriminator, build_stylegan, LATENT_DIM, INITIAL_SIZE
from dataset import load_and_preprocess_adni, generate_random_inputs
import wandb
import os
import datetime

# Define training parameters
BATCH_SIZE = 16
EPOCHS = 100
TARGET_SIZE = (128, 128)

# Directories for AD and NC classes
ad_dirs = ['/home/groups/comp3710/ADNI/AD_NC/train/AD', '/home/groups/comp3710/ADNI/AD_NC/test/AD']

# Initialize Weights and Biases for each model
def init_wandb(class_name):
    print(f"Initializing Weights and Biases for {class_name}...")
    wandb.init(
        project=f"StyleGAN ADNI {class_name} Test", 
        entity="samwolfenden-university-of-queensland",
        config={
            "gen learning rate": 0.00002,
            "disc learning rate": 0.00001,
            "epochs": EPOCHS,
            "optimizer": "Adam",
            "scheduler": "ExponentialDecay",
            "cross entropy loss": "BinaryCrossentropy",
            "image size": TARGET_SIZE,
            "batch size": BATCH_SIZE,
            "class_name": class_name
        },
        save_code=True
    )

# Train Step function
@tf.function
def train_step(real_images, generator, discriminator, generator_optimizer, discriminator_optimizer):
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
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)
        real_loss = cross_entropy(tf.fill(tf.shape(real_output), 0.9), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = real_loss + fake_loss

    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Function to set up the models and optimizers for AD dataset
def setup_ad_models():
    print("Loading AD models...")
    generator = build_generator()
    discriminator = build_discriminator()

    # Learning rate schedules
    generator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.00002, decay_steps=1000, decay_rate=1.01, staircase=True
    )
    discriminator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.00001, decay_steps=1000, decay_rate=0.99, staircase=True
    )

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=generator_lr_schedule, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=discriminator_lr_schedule, beta_1=0.5)

    # Call the models once to ensure variables are initialized before training
    latent_vectors, constant_inputs = generate_random_inputs(BATCH_SIZE, LATENT_DIM, INITIAL_SIZE)
    generator([latent_vectors, constant_inputs])
    discriminator(tf.random.normal([BATCH_SIZE] + list(TARGET_SIZE) + [1]))

    print("AD models compiled.")
    
    return generator, discriminator, generator_optimizer, discriminator_optimizer

# Function to generate and save t-SNE embeddings
def generate_tsne_embeddings(model, class_name):
    # Generate latent vectors and corresponding images
    latent_vectors, constant_inputs = generate_random_inputs(100, LATENT_DIM, INITIAL_SIZE)
    generated_images = model([latent_vectors, constant_inputs], training=False)
    flattened_images = np.reshape(generated_images, (100, -1))  # Flatten images for t-SNE
    
    # Apply t-SNE on the latent vectors
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(flattened_images)

    # Plot t-SNE results
    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=np.arange(100), cmap='viridis', s=10)
    plt.title(f"t-SNE embeddings for {class_name} at latent space")
    plt.colorbar()
    plt.savefig(f'tsne_{class_name}.png')
    plt.close()
    print(f"t-SNE plot saved for {class_name}")

# Train function for each class
def train(epochs, dataset, class_name, setup_func):
    generator, discriminator, generator_optimizer, discriminator_optimizer = setup_func()
    init_wandb(class_name)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} for {class_name}...")

        for batch in dataset:
            gen_loss, disc_loss = train_step(batch, generator, discriminator, generator_optimizer, discriminator_optimizer)

        print(f"Epoch {epoch + 1} completed.")
        print(f"{class_name} Generator Loss", gen_loss.numpy(), "Discriminator Loss", disc_loss.numpy())

        # Log losses to Weights and Biases
        wandb.log({"Generator Loss": float(gen_loss), "Discriminator Loss": float(disc_loss)})

        # Save images and models periodically
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, class_name)
            generate_tsne_embeddings(generator, class_name)

        if (epoch + 1) == epochs:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            generator.save(f'generator_model_{class_name}_time_{timestamp}.h5')

    wandb.finish()

def generate_and_save_images(model, epoch, class_name):
    latent_vectors, constant_inputs = generate_random_inputs(16, LATENT_DIM, INITIAL_SIZE)
    predictions = model([latent_vectors, constant_inputs], training=False)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)

        if predictions.shape[-1] == 1:
            plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        else:
            plt.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')

    # Save the generated image grid
    image_filename = f'StyleGAN_{class_name}_image_at_epoch_{epoch:04d}.png'
    plt.savefig(image_filename)
    plt.close()

    print(f"Saved image for {class_name} at epoch {epoch}: {image_filename}")

# Main function to train on both AD and NC datasets
if __name__ == "__main__":
    cross_entropy = tf.keras.losses.BinaryCrossentropy()

    # Load and preprocess the AD and NC datasets
    ad_dataset = load_and_preprocess_adni(ad_dirs, TARGET_SIZE, BATCH_SIZE)

    # Train separate models for AD and NC
    print("Training on AD dataset...")
    train(EPOCHS, ad_dataset, "AD", setup_ad_models)
