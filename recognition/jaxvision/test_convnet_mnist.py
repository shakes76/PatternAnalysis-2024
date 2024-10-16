'''
Test MNIST classifications with JAX and Haiku
'''
import haiku as hk
import jax
import numpy as np
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds #stable
from typing import Mapping, Tuple, NamedTuple
import pickle

#locals
import filenames
import preprocess
import networks

print(jax.devices())
print("Using", jnp.ones(3).device_buffer.device())

#TFDS returns dict of image and labels
Batch = Mapping[str, np.ndarray]
class TrainState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState

#===========
#parameters
epochs = 3500
batch_size = 128
channels = 64
tfdtype = tf.float32
train_set = 'train'
download = True

#===========
#paths and names
home_dir = 'I:/'
# home_dir = '/home/shakes/'
model_name = "ConvNet"
logs_name = 'logs'

#network IO paths
path = home_dir+'data/mnist/training/'
operation_type = "classify_"+model_name+"_mnist"
print("Summary: Classify for MNIST images")

#Check output paths exist, if not create
output_path = path+"output_"+operation_type+"/"
filenames.createDirectory(output_path)

#===========
#helper functions
rng_preproc = tf.random.Generator.from_seed(123, alg='philox')
# A wrapper function for updating seeds
def f(item):
    seed = rng_preproc.make_seeds(2)[0]
    return preprocess.preprocess_samplewise(item, seed)

#===========
#load data
train_ds, info = tfds.load('mnist', split=train_set, shuffle_files=True, data_dir=home_dir+'data', download=download, with_info=True)
total_images = info.splits[train_set].num_examples
total_batches = total_images//batch_size
total_steps = total_batches*epochs
xSize, ySize, rgbSize = info.features['image'].shape
num_classes = info.features['label'].num_classes
get_label_name = info.features['label'].int2str
print("Found", total_images, "training images")
print("No. of Classes:", num_classes)
print("Image Size:", info.features['image'].shape)
print(info)

#training set
train_ds = train_ds.cache().repeat()
train_ds = train_ds.shuffle(10*total_batches)
train_ds = train_ds.map(f, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
train_ds = iter(tfds.as_numpy(train_ds))

#testing set
test_ds = tfds.load('mnist', split='test', shuffle_files=False, data_dir=home_dir+'data', download=download, with_info=False)
test_ds = test_ds.cache().repeat()
test_ds = test_ds.map(preprocess.normalize_samplewise, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(batch_size)
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
test_ds = iter(tfds.as_numpy(test_ds))

class ConvNet(hk.Module):
    '''
    A standard VGG-like network but much smaller
    Assumes image is input and returns logits
    conv_layers defines this many conv layers
    channels sets the initial number of filters
    Subsequent conv layers are increased by powers of 2
    dense_size sets the largest dense layer size right after conv layers
    and uses two dense layers down (each downsampled by 4) to output_size
    '''
    def __init__(self, output_size, conv_layers=2, kernel_shape=3, channels=32, dense_size=256, bn_config=None, logits_config=None, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.conv_layers = conv_layers
        self.kernel_shape = kernel_shape
        self.channels = channels
        self.dense_size = dense_size
        self.bn_config = dict(bn_config or {})

        #batch norm config
        self.bn_config.setdefault("decay_rate", 0.9)
        self.bn_config.setdefault("eps", 1e-5)
        self.bn_config.setdefault("create_scale", True)
        self.bn_config.setdefault("create_offset", True)

        self.logits_config = dict(logits_config or {})
        self.logits_config.setdefault("w_init", jax.numpy.zeros)
        self.logits_config.setdefault("name", "logits")

    def __call__(self, x, is_training, test_local_stats=False):
        '''
        ConvNet Network
        '''
        #define conv layers
        net1=networks.NormBlock(self.channels, stride=1, rate=1, kernel_shape=3, bn_config=self.bn_config, name="normblock_initial_1")(x, is_training)
        net2=networks.NormBlock(self.channels, stride=1, rate=1, kernel_shape=7, bn_config=self.bn_config, name="normblock_initial_2")(x, is_training)
        net3=networks.NormBlock(self.channels, stride=1, rate=1, kernel_shape=11, bn_config=self.bn_config, name="normblock_initial_3")(x, is_training)
        net4=networks.NormBlock(self.channels, stride=1, rate=1, kernel_shape=15, bn_config=self.bn_config, name="normblock_initial_4")(x, is_training)
        net5=networks.NormBlock(self.channels, stride=1, rate=1, kernel_shape=19, bn_config=self.bn_config, name="normblock_initial_5")(x, is_training)
        # net6=networks.NormBlock(self.channels, stride=1, rate=1, kernel_shape=23, bn_config=self.bn_config, name="normblock_initial_6")(x, is_training)
        for n in range(1,self.conv_layers):
            net1=networks.NormBlock(self.channels, stride=1, rate=1, kernel_shape=3, bn_config=self.bn_config, name="normblock_"+str(n)+"_x1")(net1, is_training)
            net2=networks.NormBlock(self.channels, stride=1, rate=1, kernel_shape=7, bn_config=self.bn_config, name="normblock_"+str(n)+"_x2")(net2, is_training)
            net3=networks.NormBlock(self.channels, stride=1, rate=1, kernel_shape=11, bn_config=self.bn_config, name="normblock_"+str(n)+"_x3")(net3, is_training)
            net4=networks.NormBlock(self.channels, stride=1, rate=1, kernel_shape=15, bn_config=self.bn_config, name="normblock_"+str(n)+"_x4")(net4, is_training)
            net5=networks.NormBlock(self.channels, stride=1, rate=1, kernel_shape=19, bn_config=self.bn_config, name="normblock_"+str(n)+"_x5")(net5, is_training)
            # net6=networks.NormBlock(self.channels, stride=1, rate=1, kernel_shape=23, bn_config=self.bn_config, name="normblock_"+str(n)+"_x6")(net6, is_training)
        net=jax.numpy.concatenate([net1, net2, net3, net4, net5], axis=-1)
        # GlobalAveragePooling2D
        net=jax.numpy.mean(net, axis=[1, 2])
        #define dense layers
        # net=hk.Flatten()(net)
        net=hk.Linear(self.dense_size)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(self.dense_size//4)(net)
        net=jax.nn.relu(net)
        net=hk.Linear(self.output_size, **self.logits_config)(net)

        return net

#===========
##create model
def _forward(batch: Batch, is_training: bool) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    # cnn = networks.ConvNet(num_classes, conv_layers=3, kernel_shape=3, channels=channels, dense_size=128)
    cnn = ConvNet(num_classes, conv_layers=1, kernel_shape=3, channels=channels, dense_size=128)
    return cnn(batch["image"], is_training=is_training)

# Make the network and optimiser.
network = hk.transform_with_state(_forward)
opt = optax.adam(1e-3)

#loss function
def loss_fn(params: hk.Params, state: hk.State, batch: Batch) -> jnp.ndarray:
    '''
    Cross entropy including L2 weight regularisation
    '''
    logits, state = network.apply(params, state, None, batch, is_training=True)
    labels = jax.nn.one_hot(batch["label"], num_classes)

    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]

    loss = softmax_xent + 1e-4 * l2_loss

    return loss, (loss, state)

# Evaluation metric (classification accuracy).
@jax.jit
def accuracy(params: hk.Params, state: hk.State, batch: Batch) -> jnp.ndarray:
    logits, _ = network.apply(params, state, None, batch, is_training=False)
    return jnp.mean(jnp.argmax(logits, axis=-1) == batch["label"])

@jax.jit
def update(
        train_state: TrainState,
        batch: Batch,
    ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    params, state, opt_state = train_state
    grads, (loss, new_state) = jax.grad(loss_fn, has_aux=True)(params, state, batch)
    updates, new_opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    train_state = TrainState(new_params, new_state, new_opt_state)
    return train_state, loss

# Initialization requires an example input.
batch = next(train_ds)

# For initialization we need the same random key on each device.
rng = jax.random.PRNGKey(42)
# Initialize network and optimiser; note we draw an input to get shapes.
# params = avg_params = network.init(jax.random.PRNGKey(42), next(train_ds))
def initial_state(rng: jnp.ndarray, batch: Batch) -> TrainState:
    """Computes the initial network state."""
    params, state = network.init(rng, batch, is_training=True)
    opt_state = opt.init(params)
    return TrainState(params, state, opt_state)
train_state = initial_state(rng, batch)
params, state, opt_state = train_state

# Print a useful summary of the execution of our module.
summary = hk.experimental.tabulate(accuracy, 
                            columns=('module','output','params_size','params_bytes'), 
                            filters=('has_output','has_params'))(params, state, batch)
print(summary)

# Train/eval loop.
for step in range(epochs):
    if step % 100 == 0:
        # Periodically evaluate classification accuracy on train & test sets.
        train_accuracy = accuracy(params, state, next(train_ds))
        # train_accuracy = accuracy(avg_params, train_ds)
        test_accuracy = accuracy(params, state, next(test_ds))
        # test_accuracy = accuracy(avg_params, test_ds)
        train_accuracy, test_accuracy = jax.device_get(
            (train_accuracy, test_accuracy))
        print(f"[Step {step}] Train / Test accuracy: "
                f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

    # Do SGD on a batch of training examples.
    train_state, loss = update(train_state, next(train_ds))
    params, state, opt_state = train_state

#test eval
# NOTE: We use `jit` not `pmap` here because we want to ensure that we see all
# eval data once and this is not easily satisfiable with pmap (e.g. n=3).
@jax.jit
def eval_batch(
    params: hk.Params,
    state: hk.State,
    batch: Batch,
) -> jnp.ndarray:
    """Evaluates a batch."""
    logits, _ = network.apply(params, state, None, batch, is_training=False)
    predicted_label = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(jnp.equal(predicted_label, batch['label']))
    return correct.astype(jnp.float32)

#save the model
pickle.dump(params, open(filenames.os.path.join(path, "mnist_convnet.pkl"), "wb"))

#testing set
test_ds = tfds.load('mnist', split='test', shuffle_files=False, data_dir=home_dir+'data', download=download, with_info=False)
total_images = info.splits['test'].num_examples
test_ds = test_ds.map(preprocess.normalize_samplewise, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(batch_size)
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
test_ds = tfds.as_numpy(test_ds)

correct = jnp.array(0)
total = 0
for batch in test_ds:
    # print(f"Total {total}: Test batch shape is {batch['image'].shape}")
    correct += eval_batch(params, state, batch)
    total += batch['label'].shape[0]

assert total == total_images, total
print('top_1_acc:', correct.item() / total)

print("END")