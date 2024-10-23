from keras.optimizers import Adam
import matplotlib.pyplot as plt

from dataset import get_training_data, data_loader
from modules import unet_2d, dice_coef_prostate, total_loss

def train(model, train_generator, test_generator, steps_per_epoch, val_steps_per_epoch, epochs):

    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=[total_loss], metrics=['accuracy', dice_coef_prostate])
    model.summary()

    history = model.fit(train_generator,
                        validation_data=test_generator,
                        epochs=epochs, steps_per_epoch=steps_per_epoch,
                        validation_steps=val_steps_per_epoch)

    model.save('models/model.h5')

    # plot training and validation loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Dice loss + focal loss')
    plt.legend()
    plt.savefig('plots/dice_loss.png')
    plt.show()

    # plot prostate training and validation loss
    loss = history.history['dice_coef_prostate']
    val_loss = history.history['val_dice_coef_prostate']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training coef prostate')
    plt.plot(epochs, val_loss, 'r', label='Validation coef prostate')
    plt.xlabel('Epochs')
    plt.ylabel('Dice similarity')
    plt.legend()
    plt.savefig('plots/dice_coef_prostate.png')
    plt.show()

    # plot training and validation accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('plots/accuracy.png')
    plt.show()


# Driver code
if __name__ == '__main__':
    model = unet_2d(output_classes=6, input_shape=(256, 128, 1))

    # fetch training data
    train_data, test_data = get_training_data(image_limit=None)

    BATCH_SIZE = 4
    EPOCHS = 15
    STEPS_PER_EPOCH = len(train_data[0])//BATCH_SIZE
    VAL_STEPS_PER_EPOCH = len(test_data[0])//BATCH_SIZE

    # build data generators 
    train_generator, test_generator = data_loader(train_data, test_data, batch_size=BATCH_SIZE)

    # train the U-Net
    train(model, train_generator, test_generator, steps_per_epoch=STEPS_PER_EPOCH,
          val_steps_per_epoch=VAL_STEPS_PER_EPOCH, epochs=EPOCHS)
