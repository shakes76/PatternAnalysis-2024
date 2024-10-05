# train.py
import tensorflow as tf
from modules import define_discriminator, discriminator_loss, generator_loss, StyleGANGenerator, MappingNetwork#mapping_network, StyleGANGenerator#define_generator
from dataset import load_data
import numpy as np
from matplotlib import pyplot as plt

# Define hyperparameters
batch_size = 10
latent_dim = 512
epochs = 1
mixing_prob = 0.90

# Initialize models
discriminator = define_discriminator()
mapping_network = MappingNetwork()  # Use the modified mapping network
generator = StyleGANGenerator(mapping_network)

# Define optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.99, epsilon=1e-8) #TODO: Paper also suggests beta_1=0, beta_2=0.99, epsilon=1e-8 for all learning rates.

# TODO: ADD OPTIMISER FOR MAPPING NETWORK HERE.
mapping_network_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.0, beta_2=0.99, epsilon=1e-8) # Papers suggest 100x lower, I will do 10x because my learning rates are already very low.

# Define training step
@tf.function
def train_step(images):
    noise1 = tf.random.normal([batch_size, latent_dim])
    noise2 = tf.random.normal([batch_size, latent_dim])
    
    # Need a persistent gradient tape as we will be using it to update both the generator and the mapping network
    # without it being persistent it will be "used" after being used for the generator.
    with tf.GradientTape(persistent=True) as gen_tape, tf.GradientTape() as disc_tape:

        # Generating two distinct subsections of style space from noise in order to 
        # use style mixing.
        style1 = mapping_network(noise1)
        style2 = mapping_network(noise2)

        #print("BEFORE GENERATOR WITH style1 = ", type(style1))
        #print("BEFORE GENERATOR WITH style2 = ", type(style2))

        generated_images = generator(style1, style2, mixing_prob=mixing_prob)
        
        #print("AFTER GENERATOR")

        #print("generated images shape =", generated_images.shape)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        
        gen_loss = generator_loss(fake_output) #TODO: CAN POSSIBLY ADD PATH_LENGTH_REGULARISATION TO THIS
        disc_loss = discriminator_loss(real_output, fake_output) #TODO: CAN POSSIBLY ADD R1 REGULARISATION TO THIS.
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Update mapping network separately
    mapping_network_variables = mapping_network.trainable_variables
    gradients_of_mapping_network = gen_tape.gradient(gen_loss, mapping_network_variables)
    mapping_network_optimizer.apply_gradients(zip(gradients_of_mapping_network, mapping_network_variables))

    # Check for gradient issues - trying to fix the warning that I am having.
    for var, grad in zip(generator.trainable_variables, gradients_of_generator):
        if grad is None:
            print(f"Warning: Gradient for generator variable {var.name} is None")
    
    for var, grad in zip(discriminator.trainable_variables, gradients_of_discriminator):
        if grad is None:
            print(f"Warning: Gradient for discriminator variable {var.name} is None")

    for var, grad in zip(mapping_network_variables, gradients_of_mapping_network):
        if grad is None:
            print(f"Warning: Gradient for mapping network variable {var.name} is None")

    # So that it doesn't persist beyond its required usage (ie; into other epochs)
    del gen_tape
    
    return disc_loss, gen_loss

# Define training loop
def train():
    # Load data
    images = load_data()
    
    # Prepare dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(len(images)).batch(batch_size)
    
    print("train dataset size =", len(list(train_dataset)))

    for epoch in range(epochs):
        for i, image_batch in enumerate(train_dataset):
            #print("CALLING TRAIN STEP")
            print("Handling batch ", i)
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
