import tensorflow as tf
import matplotlib.pyplot as plt
from modules import build_generator, build_discriminator, build_stylegan, LATENT_DIM, INITIAL_SIZE, FINAL_SIZE
from dataset import create_adni_dataset, generate_random_inputs
import numpy as np
import wandb

# Define training parameters
BATCH_SIZE = 16
EPOCHS = 30
TARGET_SIZE = (256, 256)  # Change target size to 256x256

# Define initial learning rate and decay parameters for the generator
initial_learning_rate = 0.0001
decay_steps = 100 
decay_rate = 1.05  

# Create a learning rate schedule for the generator
generator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)

# Define initial learning rate and decay parameters for the discriminator
initial_discriminator_learning_rate = 0.00001  
discriminator_decay_steps = 100 
discriminator_decay_rate = 0.95

# Create a learning rate schedule for the discriminator
discriminator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_discriminator_learning_rate,
    decay_steps=discriminator_decay_steps,
    decay_rate=discriminator_decay_rate,
    staircase=True
)

# Load the models
print("Loading models...")
generator = build_generator()
print("Generator model loaded.")
discriminator = build_discriminator()
print("Discriminator model loaded.")
stylegan = build_stylegan()
print("StyleGAN model loaded.")

# Compile the models with the learning rate schedule for the generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=generator_lr_schedule, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=discriminator_lr_schedule, beta_1=0.5)

print("Models compiled.")

# Initialize Weights and Biases
print("Initializing Weights and Biases...")
wandb.init(
    project="StyleGAN ADNI Test", 
    entity="samwolfenden-university-of-queensland",
    config={
        "gen learning rate": initial_learning_rate,
        "disc learning rate": initial_discriminator_learning_rate,
        "epochs": EPOCHS,
        "optimizer": type(generator_optimizer).__name__,
        "scheduler": "ExponentialDecay",
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
def train_step(real_images, update_counter):
    latent_vectors, constant_inputs = generate_random_inputs(BATCH_SIZE, LATENT_DIM, INITIAL_SIZE)

    # Train generator
    with tf.GradientTape() as gen_tape:
        generated_images = generator([latent_vectors, constant_inputs], training=True)
        fake_output = discriminator(generated_images, training=True)
        
        # Adversarial loss for generator
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)  # No smoothing for generator

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))

    # Update discriminator only if the counter reaches 4
    if update_counter % 16 == 0:
        with tf.GradientTape() as disc_tape:
            real_output = discriminator(real_images, training=True)
            fake_output = discriminator(generated_images, training=True)

            real_loss = cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss

        disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

        return gen_loss, disc_loss
    else:
        return gen_loss, None  # Return None for disc_loss if not updated

def train(epochs):
    print("Starting training...")
    update_counter = 0  # Initialize the update counter
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}...")
        for step, real_images in enumerate(adni_dataset):
            gen_loss, disc_loss = train_step(real_images, update_counter)  # Pass the update counter
            
            # Log losses to Weights and Biases
            wandb.log({
                "Generator Loss": gen_loss.numpy(),
                "Discriminator Loss": disc_loss.numpy() if disc_loss is not None else None
            })
            
            update_counter += 1  # Increment the update counter
        
        print(f"Epoch {epoch + 1} completed.")
        print("Generator Loss", gen_loss.numpy(), "Discriminator Loss", disc_loss.numpy() if disc_loss is not None else None)

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