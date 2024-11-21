'''
Preprocessing helper functions
'''
import tensorflow as tf

#constants
cifar_mean = tf.constant([[[0.49139968, 0.48215841, 0.44653091]]])
cifar_std = tf.constant([[[0.24703223, 0.24348513, 0.26158784]]])

@tf.function
def normalize_cifar(item, dtype=tf.float32):
    image = tf.cast(item['image'], dtype) / 255.0
    new_item = {}
    new_item['image'] = (image - cifar_mean) / cifar_std
    new_item['label'] = item['label']
    return new_item

@tf.function
def normalize_samplewise(item, dtype=tf.float32):
    image = tf.cast(item['image'], dtype) / 255.0
    new_item = {}
    new_item['image'] = (image - tf.math.reduce_mean(image) ) / tf.math.reduce_std(image)
    new_item['label'] = item['label']
    return new_item

@tf.function
def preprocess_samplewise(item, seed):
    '''
    Preprocess the image with sample wise stats, seed unused
    '''
    return normalize_samplewise(item)

@tf.function
def preprocess_cifar(item, seed, crop_pad=4, cutout_size=None):
    '''
    Preprocess the image with CIFAR stats and standard preprocessing:
    Random Flip L/R
    Pad by 4 pixels (relfect) with 32x32 random crops
    '''
    new_item = normalize_cifar(item)

    # Make a new seed
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

    image = tf.image.stateless_random_flip_left_right(new_item['image'], new_seed) #horizontal flip

    image_shape = tf.shape(image)
    image = tf.pad(image, [[crop_pad, crop_pad], [crop_pad, crop_pad], [0, 0]], mode='REFLECT') #pad
    image = tf.image.stateless_random_crop(image, size=image_shape, seed=seed)

    new_item['image'] = image
    if cutout_size is not None:
        mask = tf.ones([cutout_size, cutout_size], dtype=tf.int32)
        start = tf.random_uniform([2], minval=0, maxval=32, dtype=tf.int32)
        mask = tf.pad(mask, [[cutout_size + start[0], 32 - start[0]],
                            [cutout_size + start[1], 32 - start[1]]])
        mask = mask[cutout_size: cutout_size + 32,
                    cutout_size: cutout_size + 32]
        mask = tf.reshape(mask, [32, 32, 1])
        mask = tf.tile(mask, [1, 1, 3])
        new_item['image'] = tf.where(tf.equal(mask, 0), x=image, y=tf.zeros_like(image))

    return new_item
