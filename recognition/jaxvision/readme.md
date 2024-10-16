# JAX Vision
This is a simple library for doing vision tasks using JAX and Haiku.

There are various baseline networks for MNIST and CIFAR classification tasks, such as the ConvNet (VGG-like) and ResNet-18.

The main module is the network module, which contains the wrapped layers and various networks using the Haiku interface.

## Performance
The ResNet-18 (based on the original Haiku ResNet examples) achieves accuracy 0.94 consistently.

## Installing JAX/Haiku 
See the install.md file for details.
