import tensorflow as tf
import matplotlib.pyplot as plt
from modules import build_generator, build_discriminator, build_stylegan, LATENT_DIM, INITIAL_SIZE
from dataset import create_adni_dataset, generate_random_inputs
import wandb


# Define training parameters
BATCH_SIZE = 16
EPOCHS = 170
TARGET_SIZE = (128, 128)  # Change target size to 128x128

# Load the models
print("Loading models...")
generator = build_generator()
print("Generator model loaded.")
discriminator = build_discriminator()
print("Discriminator model loaded.")
stylegan = build_stylegan()
print("StyleGAN model loaded.")

# Define initial learning rate and decay parameters for the generator
initial_learning_rate = 0.00002
decay_steps = 1000
decay_rate = 1.01

# Create a learning rate schedule for the generator
generator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)

# Define initial learning rate and decay parameters for the discriminator
initial_discriminator_learning_rate = 0.00001
discriminator_decay_steps = 1000
discriminator_decay_rate = 0.99

# Create a learning rate schedule for the discriminator
discriminator_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_discriminator_learning_rate,
    decay_steps=discriminator_decay_steps,
    decay_rate=discriminator_decay_rate,
    staircase=True
)

# Compile the models
print("Compiling models...")
generator_optimizer = tf.keras.optimizers.experimental.Adam(learning_rate=generator_lr_schedule, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.experimental.Adam(learning_rate=discriminator_lr_schedule, beta_1=0.5)

print("Models compiled.")

# Initialize Weights and Biases
print("Initializing Weights and Biases...")
wandb.init(
    project="StyleGAN ADNI Test", 
    entity="samwolfenden-university-of-queensland",
    config={
        "gen learning rate": generator_lr_schedule,
        "disc learning rate": discriminator_lr_schedule,
        "epochs": EPOCHS,
        "optimizer": type(generator_optimizer).__name__,
        "scheduler": type(discriminator_optimizer).__name__,
        "cross entropy loss": type(tf.keras.losses.BinaryCrossentropy()).__name__,
        "name": "SD-ADNI - VAE and Unet",
        "image size": TARGET_SIZE,
        "batch size": BATCH_SIZE
    },
    save_code=True
)

# Load the ADNI dataset
print("Loading ADNI dataset...")
adni_dataset = create_adni_dataset(BATCH_SIZE, TARGET_SIZE)
print("ADNI dataset loaded.")

# Define the loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy()

# Define label smoothing values
REAL_LABEL_SMOOTHING = 0.9

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
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Apply label smoothing for real and fake labels
        real_loss = cross_entropy(tf.fill(tf.shape(real_output), REAL_LABEL_SMOOTHING), real_output)  
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
        print("Generator Loss", gen_loss.numpy(), "Discriminator Loss", disc_loss.numpy())

        # Log losses to Weights and Biases
        wandb.log({"Generator Loss": gen_loss, "Discriminator Loss": disc_loss})
        print("Logged losses to Weights and Biases.")

        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1)
        
        if (epoch + 1) == EPOCHS:
            generator.save(f'generator_model_epoch_{epoch + 1}.h5')
            discriminator.save(f'discriminator_model_epoch_{epoch + 1}.h5') 

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
    
    plt.savefig(f'StyleGAN_image_at_epoch_{epoch:04d}.png')
    plt.close()
    print(f"Images saved for epoch {epoch}.")

if __name__ == "__main__":
    gen_loss, disc_loss = train(EPOCHS)
