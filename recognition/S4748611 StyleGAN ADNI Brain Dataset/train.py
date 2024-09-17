import tensorflow as tf
import matplotlib.pyplot as plt
from modules import build_generator, build_discriminator, build_stylegan, LATENT_DIM, INITIAL_SIZE, FINAL_SIZE
from dataset import create_mnist_dataset, generate_random_inputs

# Define training parameters
BATCH_SIZE = 32
EPOCHS = 30

# Load the models
generator = build_generator()
discriminator = build_discriminator()
stylegan = build_stylegan()

# Load the MNIST dataset
mnist_dataset = create_mnist_dataset(BATCH_SIZE)

# Compile the models
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.5)

# Define the loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy()

@tf.function
def train_step(real_images):
    latent_vectors, constant_inputs = generate_random_inputs(BATCH_SIZE, LATENT_DIM, INITIAL_SIZE)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([latent_vectors, constant_inputs], training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = real_loss + fake_loss

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(epochs):
    for epoch in range(epochs):
        for batch in mnist_dataset:
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

if __name__ == "__main__":
    train(EPOCHS)

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