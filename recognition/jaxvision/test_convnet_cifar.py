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
import time

#prevent TF from hogging GPU memory in conflict with XLA and JAX
tf.config.experimental.set_visible_devices([], "GPU")

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

#parameters
epochs = 34
batch_size = 128
tfdtype = tf.float32
train_set = 'train'
download = True

#===========
#paths and names
home_dir = '/home/shakes/'
model_name = "ConvNet"
logs_name = 'logs'

#network IO paths
path = home_dir+'data/mnist/training/'
operation_type = "classify_"+model_name+"_cifar"
print("Summary: Classify CIFAR images")

#Check output paths exist, if not create
output_path = path+"output_"+operation_type+"/"
filenames.createDirectory(output_path)

#===========
#helper functions
rng_preproc = tf.random.Generator.from_seed(123, alg='philox')
# A wrapper function for updating seeds
def f(item):
    seed = rng_preproc.make_seeds(2)[0]
    return preprocess.preprocess_cifar(item, seed)

#===========
#load data
train_ds, info = tfds.load('cifar10', split=train_set, shuffle_files=True, data_dir=home_dir+'data', download=download, with_info=True)
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
test_ds = tfds.load('cifar10', split='test', shuffle_files=False, data_dir=home_dir+'data', download=download, with_info=False)
test_ds = test_ds.cache().repeat()
test_ds = test_ds.map(preprocess.normalize_cifar, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(batch_size)
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
test_ds = iter(tfds.as_numpy(test_ds))

#===========
##create model
def _forward(batch: Batch, is_training: bool) -> jnp.ndarray:
    '''
    Forward pass through network
    '''
    cnn = networks.ConvNet(num_classes, conv_layers=3, kernel_size=3, channels=128, dense_size=1024)
    return cnn(batch["image"], is_training=is_training)

# Make the network and optimiser.
network = hk.transform_with_state(_forward)

#setup optimizer
total_batch_size = batch_size*jax.device_count()
num_train_steps = ( (total_images*epochs)//total_batch_size )
# lr_schedule = optax.piecewise_interpolate_schedule('linear', 0.1, {0: 0, 15: 1., 30: 0.05, 35: 0})
lr_schedule = optax.linear_onecycle_schedule(
        transition_steps=num_train_steps,
        peak_value=0.1,
        pct_start=15/35.,
        pct_final=30/35.,
        div_factor=20.,
        final_div_factor=200.)
opt = optax.sgd(lr_schedule, momentum=0.9)
# print([lr_schedule(step) for step in range(10)])

#loss function
def loss_fn(params: hk.Params, state: hk.State, batch: Batch) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, hk.State]]:
    '''
    Cross entropy including L2 weight regularisation
    '''
    logits, state = network.apply(params, state, None, batch, is_training=True)
    labels = jax.nn.one_hot(batch["label"], num_classes)

    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]

    loss = softmax_xent + 5e-4 * l2_loss

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

# For initialization we need the same random key on each device.
rng = jax.random.PRNGKey(42)
# rng = jnp.broadcast_to(rng, (jax.local_device_count(),) + rng.shape)
# Initialize network and optimiser; note we draw an input to get shapes.
# params = avg_params = network.init(jax.random.PRNGKey(42), next(train_ds))
def initial_state(rng: jnp.ndarray, batch: Batch) -> TrainState:
    """Computes the initial network state."""
    params, state = network.init(rng, batch, is_training=True)
    opt_state = opt.init(params)
    return TrainState(params, state, opt_state)

# Initialization requires an example input.
batch = next(train_ds)
# train_state = jax.pmap(initial_state)(rng, batch)
train_state = initial_state(rng, batch)
params, state, opt_state = train_state

# Print a useful summary of the execution of our module.
summary = hk.experimental.tabulate(update, 
                            columns=('module','output','params_size','params_bytes'), 
                            filters=('has_output','has_params'))(train_state, batch)
print(summary)

# Train/eval loop.
loss = 999.
start = time.time() #time generation
for step in range(num_train_steps):
    if step % 100 == 0:
        # Periodically evaluate classification accuracy on train & test sets.
        train_accuracy = accuracy(params, state, next(train_ds))
        test_accuracy = accuracy(params, state, next(test_ds))
        train_accuracy, test_accuracy = jax.device_get(
            (train_accuracy, test_accuracy))
        print(f"[Step {step}, Loss {loss:.5f}] Train / Test accuracy: "
                f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

    # Optimize on a batch of training examples.
    train_state, loss = update(train_state, next(train_ds))
    params, state, opt_state = train_state
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

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

#testing set
test_ds = tfds.load('cifar10', split='test', shuffle_files=False, data_dir=home_dir+'data', download=download, with_info=False)
total_images = info.splits['test'].num_examples
test_ds = test_ds.map(preprocess.normalize_cifar, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(batch_size)
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
test_ds = tfds.as_numpy(test_ds)

correct = jnp.array(0)
total = 0
start = time.time() #time generation
for batch in test_ds:
    # print(f"Total {total}: Test batch shape is {batch['image'].shape}")
    correct += eval_batch(params, state, batch)
    total += batch['label'].shape[0]
end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

assert total == total_images, total
print('top_1_acc:', correct.item() / total)

print("END")