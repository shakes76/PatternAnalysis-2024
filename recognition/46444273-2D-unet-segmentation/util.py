import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_images_labels(images, masks):
    plt.figure(figsize=(10, 10))
    for i in range(16):
      if i % 2 == 0:
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(np.array(images[i, :, :]))
        plt.axis("off")
        mask = tf.argmax(masks[i, :, :, :], axis=2)
        ax = plt.subplot(4, 4, i + 2)
        plt.imshow(np.array(mask))
        plt.axis("off")
    plt.show()

def plot_index(idx, images, masks):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(images[idx])
    plt.subplot(122)
    mask = tf.argmax(masks[idx], axis=2)
    plt.imshow(np.array(mask))
    plt.show()
