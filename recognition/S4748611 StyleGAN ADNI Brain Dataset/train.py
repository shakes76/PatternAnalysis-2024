import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from modules import build_generator, build_discriminator, LATENT_DIM, INITIAL_SIZE
from dataset import load_and_preprocess_adni, generate_random_inputs
import wandb

tf.config.experimental_run_functions_eagerly(True)  # Disable the experimental optimizer


# Define training parameters
BATCH_SIZE = 16
EPOCHS = 100
TARGET_SIZE = (128, 128)

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

@tf.function
def train_step(real_images, generator, discriminator, generator_optimizer, discriminator_optimizer):
    latent_vectors, constant_inputs = generate_random_inputs(BATCH_SIZE, LATENT_DIM, INITIAL_SIZE)

    cross_entropy = tf.keras.losses.BinaryCrossentropy()


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

def generate_tsne_embeddings(model, class_name):
    latent_vectors, constant_inputs = generate_random_inputs(100, LATENT_DIM, INITIAL_SIZE)
    generated_images = model([latent_vectors, constant_inputs], training=False)
    flattened_images = np.reshape(generated_images, (100, -1))  

    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(flattened_images)

    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=np.arange(100), cmap='viridis', s=10)
    plt.title(f"t-SNE embeddings for {class_name} at latent space")
    plt.colorbar()
    plt.savefig(f'tsne_{class_name}.png')
    plt.close()
    print(f"t-SNE plot saved for {class_name}")

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

    image_filename = f'StyleGAN_{class_name}_image_at_epoch_{epoch:04d}.png'
    plt.savefig(image_filename)
    plt.close()

    print(f"Saved image for {class_name} at epoch {epoch}: {image_filename}")

class ADModelTrainer:
    def __init__(self):
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()
        self.ad_dirs = ['/home/groups/comp3710/ADNI/AD_NC/train/AD', '/home/groups/comp3710/ADNI/AD_NC/test/AD']
        self.dataset = load_and_preprocess_adni(self.ad_dirs, TARGET_SIZE, BATCH_SIZE)
    
    def setup_models(self):
        print("Loading AD models...")
        generator = build_generator()
        discriminator = build_discriminator()

        generator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.00002, decay_steps=1000, decay_rate=1.01, staircase=True
        )
        discriminator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.00001, decay_steps=1000, decay_rate=0.99, staircase=True
        )

        generator_optimizer = Adam(learning_rate=generator_lr_schedule, beta_1=0.5)
        discriminator_optimizer = Adam(learning_rate=discriminator_lr_schedule, beta_1=0.5)

        latent_vectors, constant_inputs = generate_random_inputs(BATCH_SIZE, LATENT_DIM, INITIAL_SIZE)
        generator([latent_vectors, constant_inputs])
        discriminator(tf.random.normal([BATCH_SIZE] + list(TARGET_SIZE) + [1]))

        return generator, discriminator, generator_optimizer, discriminator_optimizer

    def train(self):
        generator, discriminator, generator_optimizer, discriminator_optimizer = self.setup_models()
        init_wandb("AD")

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS} for AD...")
            for batch in self.dataset:
                gen_loss, disc_loss = train_step(batch, generator, discriminator, generator_optimizer, discriminator_optimizer)

            print(f"Epoch {epoch + 1} completed.")
            wandb.log({"Generator Loss": float(gen_loss), "Discriminator Loss": float(disc_loss)})

            if (epoch + 1) % 10 == 0:
                generate_and_save_images(generator, epoch + 1, "AD")
                generate_tsne_embeddings(generator, "AD")

            if (epoch + 1) == EPOCHS:
                generator.save(f'generator_model_AD.h5')

        wandb.finish()

class NCModelTrainer:
    def __init__(self):
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()
        self.nc_dirs = ['/home/groups/comp3710/ADNI/AD_NC/train/NC', '/home/groups/comp3710/ADNI/AD_NC/test/NC']
        self.dataset = load_and_preprocess_adni(self.nc_dirs, TARGET_SIZE, BATCH_SIZE)
    
    def setup_models(self):
        print("Loading NC models...")
        generator = build_generator()
        discriminator = build_discriminator()

        generator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.00002, decay_steps=1000, decay_rate=1.01, staircase=True
        )
        discriminator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.00001, decay_steps=1000, decay_rate=0.99, staircase=True
        )

        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=generator_lr_schedule, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=discriminator_lr_schedule, beta_1=0.5)

        latent_vectors, constant_inputs = generate_random_inputs(BATCH_SIZE, LATENT_DIM, INITIAL_SIZE)
        generator([latent_vectors, constant_inputs])
        discriminator(tf.random.normal([BATCH_SIZE] + list(TARGET_SIZE) + [1]))

        return generator, discriminator, generator_optimizer, discriminator_optimizer

    def train(self):
        generator, discriminator, generator_optimizer, discriminator_optimizer = self.setup_models()
        init_wandb("NC")

        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS} for NC...")
            for batch in self.dataset:
                gen_loss, disc_loss = train_step(batch, generator, discriminator, generator_optimizer, discriminator_optimizer)

            print(f"Epoch {epoch + 1} completed.")
            wandb.log({"Generator Loss": float(gen_loss), "Discriminator Loss": float(disc_loss)})

            if (epoch + 1) % 10 == 0:
                generate_and_save_images(generator, epoch + 1, "NC")
                generate_tsne_embeddings(generator, "NC")

            if (epoch + 1) == EPOCHS:
                generator.save(f'generator_model_NC.h5')

        wandb.finish()

def main():
    clear_session()

    print("Starting AD Model Training...")
    ad_trainer = ADModelTrainer()
    ad_trainer.train()

    clear_session()

    print("Starting NC Model Training...")
    nc_trainer = NCModelTrainer()
    nc_trainer.train()

if __name__ == '__main__':
    main()
