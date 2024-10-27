import pathlib

from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

from dataset import get_training_data, data_loader, load_data_2D
from modules import unet_2d, dice_coef_prostate, total_loss, natural_sort_key, dice_coef
import paths

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

   # loading test data and evaluating model

    # testing images
    test_data_dir = pathlib.Path(paths.TEST_IMG_PATH).with_suffix('')
    test_image_count = len(list(test_data_dir.glob('*.nii')))
    print(f"test image count: {test_image_count}")
    # testing masks
    seg_test_data_dir = pathlib.Path(paths.TEST_LABEL_PATH).with_suffix('')
    seg_test_image_count = len(list(seg_test_data_dir.glob('*.nii')))
    print(f"seg test image count: {seg_test_image_count}")

    image_limit = None
    early_stop = False
  
    # loading train images
    test_data = list(test_data_dir.glob('*.nii'))
    test_string = [str(d) for d in test_data]
    test_string.sort(key=natural_sort_key)
    test_data = load_data_2D(test_string, normImage=False,
                              categorical=False, early_stop=early_stop,
                                image_limit=image_limit)[:image_limit,:,:]
    # loading train masks
    seg_test_data = list(seg_test_data_dir.glob('*.nii'))
    seg_test_string = [str(d) for d in seg_test_data]
    seg_test_string.sort(key=natural_sort_key)
    seg_test_data = load_data_2D(seg_test_string,
                                   normImage=False, categorical=False,
                                   early_stop=early_stop,
                                     image_limit=image_limit).astype(np.uint8)[:image_limit,:,:]

    # expand image data dims
    test_data = np.expand_dims(np.array(test_data), 3)

    # convert masks to categorical
    n_classes = 6

    from keras.utils import to_categorical
    test_labels = to_categorical(seg_test_data, num_classes=n_classes)
    test_labels = test_labels.reshape((seg_test_data.shape[0], seg_test_data.shape[1], seg_test_data.shape[2], n_classes)).astype(np.uint8)

    X_test, y_test = test_data, test_labels

    classes = ['background', 'bladder', 'body', 'bone', 'rectum', 'prostate']
    for i, c in enumerate(classes):
      y_pred = model.predict(X_test)[:,:,:,i]
      dice_score = dice_coef(y_pred, y_test[:,:,:,i])
      print(f'dice similarity for class {c} ({i}) = {round(dice_score, 3)}')


# Driver code
if __name__ == '__main__':
    model = unet_2d(output_classes=6, input_shape=(256, 128, 1))

    # fetch training and validation data
    train_data, val_data = get_training_data(image_limit=None)

    BATCH_SIZE = 4
    EPOCHS = 15
    STEPS_PER_EPOCH = len(train_data[0])//BATCH_SIZE
    VAL_STEPS_PER_EPOCH = len(val_data[0])//BATCH_SIZE

    # build data generators 
    train_generator, val_generator = data_loader(train_data, val_data, batch_size=BATCH_SIZE)

    # train the U-Net
    train(model, train_generator, val_generator, steps_per_epoch=STEPS_PER_EPOCH,
          val_steps_per_epoch=VAL_STEPS_PER_EPOCH, epochs=EPOCHS)
