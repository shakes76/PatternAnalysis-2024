from modules import UNetSegmentation
import tensorflow as tf
from dataset import load_data_2D, get_all_paths, batch_paths
import matplotlib.pyplot as plt
import numpy as np
TEST_PATH = "C:/Users/rjmah/Documents/Sem2 2024/COMP3710/testData/x_test"
TEST_SEG_PATH = "C:/Users/rjmah/Documents/Sem2 2024/COMP3710/testData/Y_test"

MODEL_PATH = "unetSeg.keras"

testPaths = get_all_paths(TEST_PATH)
testSegPaths = get_all_paths(TEST_SEG_PATH)

unet = UNetSegmentation(MODEL_PATH)

# testing
test_loss = tf.keras.metrics.Mean()

loss_fn = tf.keras.losses.Dice()

classLosses = [[], [], [], [], []] # losses for each class in testing

# batching test to reduce memory usage
x_test_batches, y_test_batches = batch_paths(testPaths, testSegPaths, 32)
for test_x_paths, test_y_paths in zip(x_test_batches, y_test_batches):
    x_test = load_data_2D(test_x_paths, normImage=True)
    y_test = load_data_2D(test_y_paths, categorical=True)

    x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

    test_logits = unet.model(x_test_tensor, training=False)
    # calculate loss for prostate class
    for i in range(5):
        PredMasks = test_logits[:, :, :, i]
        RealMasks = y_test_tensor[:, :, :, i]
        Dice = (2 * tf.reduce_sum(PredMasks * RealMasks) + 1e-6) / (tf.reduce_sum(PredMasks + RealMasks) + 1e-6)
        classLosses[i].append(Dice)
    
    test_loss_value = loss_fn(y_test_tensor, test_logits)
    test_loss.update_state(test_loss_value)

print(f"Test Loss: {test_loss.result().numpy()}")
for i in range(5):
    print(f"dice coefficient for class {i}: {np.mean(classLosses[i])}")

# load and predict 1 image
exampleImg = load_data_2D([testPaths[0]], normImage=True)
exampleSeg = load_data_2D([testSegPaths[0]])
predSeg = tf.argmax(unet.call(exampleImg, training=False)[0], axis=-1)

plt.imshow(exampleImg[0], cmap='gray')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(exampleImg[0], cmap='gray')
ax1.imshow(predSeg, alpha=0.6)
ax1.set_title("Predicted Image Mask")
ax2.imshow(exampleImg[0], cmap='gray')
ax2.imshow(exampleSeg[0], alpha=0.6)
ax2.set_title("Actual Image Mask")

plt.show()