'''
Test MNIST classifications with JAX and Haiku using an MLP

@author Shakes
'''
import jax.numpy as jnp
from jax import grad, jit, vmap, random, devices
from jax.nn import relu, one_hot, log_softmax
import optax
import tensorflow_datasets as tfds
import time

#check GPU usage
print(devices())
print("Using", jnp.ones(3).device_buffer.device())

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=0.01):
  w_key, b_key = random.split(key)
  real_w = random.normal(w_key, (n, m))
  real_b = random.normal(b_key, (n,))
  return scale * real_w, scale * real_b

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

#create params
n_targets = 10
layer_sizes = [784, 512, 512, n_targets]
num_epochs = 10
batch_size = 128
params = init_network_params(layer_sizes, random.PRNGKey(0))
print(params[0][0].dtype)

#optimizer
opt = optax.adam(1e-3)
opt_state = opt.init(params)
  
def accuracy(params, images, targets):
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
  logits = batched_predict(params, images)
  l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p, b in params) #weight decay
  softmax_xent = -jnp.mean(targets * log_softmax(logits))
  return softmax_xent + 1e-4 * l2_loss

def preprocess(x):
  x /= 255.
  # x -= x.mean()
  # x /= x.std()
  x = jnp.reshape(x, (num_pixels)) #1D sequence, vector form
  return x #real output

batch_preprocess = vmap(preprocess)

@jit
def update(params, opt_state, x, y):
  grads = grad(loss)(params, x, y) #use holomorphic=True for complex-valued loss functions
  updates, opt_state = opt.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)
  return new_params, opt_state

def predict(params, image):
  # per-example predictions
  activations = image
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = relu(outputs) #real valued relu
  
  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits

# Make a batched version of the `predict` function
batched_predict = vmap(predict, in_axes=(None, 0))

data_dir = 'I:/data'
# data_dir = '/home/shakes/data'

# Fetch full datasets for evaluation
# tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
# You can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy
mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
mnist_data = tfds.as_numpy(mnist_data)
train_data, test_data = mnist_data['train'], mnist_data['test']
num_labels = info.features['label'].num_classes
h, w, c = info.features['image'].shape
num_pixels = h * w * c

# Full train set
train_images, train_labels = train_data['image'], train_data['label']
train_images = batch_preprocess(train_images)
train_labels = one_hot(train_labels, num_labels)

# Full test set
test_images, test_labels = test_data['image'], test_data['label']
test_images = batch_preprocess(test_images)
test_labels = one_hot(test_labels, num_labels)

def get_train_batches():
  # as_supervised=True gives us the (image, label) as a tuple instead of a dict
  ds = tfds.load(name='mnist', split='train', as_supervised=True, data_dir=data_dir)
  # You can build up an arbitrary tf.data input pipeline
  ds = ds.batch(batch_size).prefetch(1)
  # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
  return tfds.as_numpy(ds)

for epoch in range(num_epochs):
  start_time = time.time()
  for x, y in get_train_batches():
    x = batch_preprocess(x)
    y = one_hot(y, num_labels)
    params, opt_state = update(params, opt_state, x, y)
  epoch_time = time.time() - start_time

  train_acc = accuracy(params, train_images, train_labels)
  test_acc = accuracy(params, test_images, test_labels)
  print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
  print("Training set accuracy {}".format(train_acc))
  print("Test set accuracy {}".format(test_acc))

print("END")
