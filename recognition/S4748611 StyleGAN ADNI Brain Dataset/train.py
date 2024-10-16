import tensorflow as tf
import matplotlib.pyplot as plt
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
nc_dirs = ['/home/groups/comp3710/ADNI/AD_NC/train/NC', '/home/groups/comp3710/ADNI/AD_NC/test/NC']

# Function to set up the models and optimizers
def setup_models():
    print("Loading models...")
    generator = build_generator()
    print("Generator model loaded.")
    discriminator = build_discriminator()
    print("Discriminator model loaded.")
    stylegan = build_stylegan()
    print("StyleGAN model loaded.")

    # Learning rate schedules
    generator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.00001, decay_steps=1000, decay_rate=1.01, staircase=True
    )
    discriminator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.00001, decay_steps=1000, decay_rate=0.99, staircase=True
    )

    generator_optimizer = tf.keras.optimizers.experimental.Adam(learning_rate=generator_lr_schedule, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.experimental.Adam(learning_rate=discriminator_lr_schedule, beta_1=0.5)

    print("Models compiled.")
    
    return generator, discriminator, generator_optimizer, discriminator_optimizer

# Initialize Weights and Biases for each model
def init_wandb(class_name):
    print(f"Initializing Weights and Biases for {class_name}...")
    wandb.init(
        project=f"StyleGAN ADNI {class_name} Test", 
        entity="samwolfenden-university-of-queensland",
        config={
            "gen learning rate": 0.00001,
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

# Train function for each class
def train(epochs, dataset, class_name):
    generator, discriminator, generator_optimizer, discriminator_optimizer = setup_models()
    init_wandb(class_name)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} for {class_name}...")
        for batch in dataset:
            gen_loss, disc_loss = train_step(batch, generator, discriminator, generator_optimizer, discriminator_optimizer)
        
        print(f"Epoch {epoch + 1} completed.")
        print(f"{class_name} Generator Loss", gen_loss.numpy(), "Discriminator Loss", disc_loss.numpy())

        # Log losses to Weights and Biases
        wandb.log({"Generator Loss": gen_loss, "Discriminator Loss": disc_loss})

        # Save images and models periodically
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, class_name)
        
        if (epoch + 1) == epochs:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            generator.save(f'generator_model_{class_name}_time_{timestamp}')
            discriminator.save(f'discriminator_model_{class_name}_time_{timestamp}')

    wandb.finish()

def generate_and_save_images(model, epoch, class_name):
    latent_vectors, constant_inputs = generate_random_inputs(16, LATENT_DIM, INITIAL_SIZE)
    predictions = model([latent_vectors, constant_inputs], training=False)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')

    img_dir = f'{class_name}_images'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    plt.savefig(f'{img_dir}/StyleGAN_{class_name}_image_at_epoch_{epoch:04d}.png')
    plt.close()


# Main function to train on both AD and NC datasets
if __name__ == "__main__":
    cross_entropy = tf.keras.losses.BinaryCrossentropy()

    # Load and preprocess the AD and NC datasets
    ad_dataset = load_and_preprocess_adni(ad_dirs, TARGET_SIZE, BATCH_SIZE)
    nc_dataset = load_and_preprocess_adni(nc_dirs, TARGET_SIZE, BATCH_SIZE)

    # Train separate models for AD and NC
    print("Training on AD dataset...")
    train(EPOCHS, ad_dataset, "AD")

    print("Training on NC dataset...")
    train(EPOCHS, nc_dataset, "NC")
