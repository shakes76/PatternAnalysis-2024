from modules import UNet
from dataset import MedicalImageDataset
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"Num GPUs Available: {len(gpus)}")
else:
    print("No GPUs available, using CPU.")


def dice_coefficient(y_true, y_pred, epsilon=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2. * intersection + epsilon) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + epsilon)
    return dice

def train_model(model, train_loader, criterion, optimizer, num_epochs=50):
    model.compile(optimizer=optimizer, loss=criterion, metrics=[dice_coefficient])
    print("> Training")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Reset the metrics at the start of the epoch
        running_loss = 0.0
        running_dice = 0.0

        for i, (images, labels) in enumerate(train_loader):
            # Perform a training step
            loss, dice_score = model.train_on_batch(images, labels)
            running_loss += loss
            running_dice += dice_score

            if (i + 1) % 100 == 0:
                print(f"Step [{i+1}/{len(train_loader)}], Loss: {loss:.5f}, Dice: {dice_score:.5f}")

        avg_loss = running_loss / len(train_loader)
        avg_dice = running_dice / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.5f}, Average Dice: {avg_dice:.5f}')

input_dims = (256, 144, 1) 
model = UNet(input_dims=input_dims)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 
criterion = tf.keras.losses.BinaryCrossentropy()

image_dir = "C:/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/HipMRI_study_keras_slices_data/keras_slices_train"
train_dataset = MedicalImageDataset(image_dir=image_dir, normImage=True, batch_size=8, shuffle=True)
dataset = train_dataset.get_dataset()

train_model(model, dataset, criterion, optimizer, num_epochs=1)
model.summary()
model.save('unet_model', save_format='tf')
