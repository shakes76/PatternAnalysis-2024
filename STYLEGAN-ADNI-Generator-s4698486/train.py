# train.py
import tensorflow as tf
from modules import define_generator, define_discriminator, discriminator_loss, generator_loss
from dataset import load_data
import numpy as np
from matplotlib import pyplot as plt

# Define hyperparameters
batch_size = 10
latent_dim = 256
epochs = 2

# Initialize models
generator = define_generator()
discriminator = define_discriminator()

# Define optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Define training step
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, latent_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return disc_loss, gen_loss

# Define training loop
def train():
    # Load data
    images = load_data()
    
    # Prepare dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(len(images)).batch(batch_size)
    
    for epoch in range(epochs):
        for image_batch in train_dataset:
            d_loss, g_loss = train_step(image_batch)
        
        print(f'Epoch {epoch+1}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}')
        
        # Visualize generated images
        if (epoch + 1) % 1 == 0:  # Save every epoch
            visualize_generated_images(epoch, generator)

def visualize_generated_images(epoch, generator, n_samples=5):
    noise = tf.random.normal([n_samples, latent_dim])
    generated_images = generator(noise, training=False)
    
    plt.figure(figsize=(25, 25))
    for i in range(n_samples):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
    plt.savefig(f'generated_images_epoch_{epoch+1}.png')
    plt.show()

if __name__ == '__main__':
    train()
