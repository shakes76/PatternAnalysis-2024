## COMP3710 Project - Topic 6 Siamese Network
## s4642448 Matt Hoffman

This repository involves documentation for a classification model built around the ISIC 2020 Kaggle Challenge dataset containing images of melanoma, specifically using a Siamese network classifier.

Siamese networks involve the use of 'two' CNNs to determine how different images are, and these networks share the same weights - which is where I assume the name comes from. In practice the same network can be used (as it shares the same weights) as long as you feed it two samples in isolation.

Following this, euclidean distance is a fairly standard way of quantifying how different the two images are.

The general process followed for building up the Siamese network is as follows:
1. Construct a Torch Dataset derivative to load the test and train sets
2. Construct a base CNN class to use as the backbone of the network
3. Implement a utility class to take in a pair of images and push them through the network
4. Build a utility class able to calculate the euclidean distance between the output results
5. Construct a utility class to find and pair images with the same labels from the dataset
6. Train the model
7. Test the model

### Test Environment and Dependencies

The training for this model was done on some of the servers I have at home. The GPUs involved were an NVIDIA GTX 1080 Ti 11GB using CUDA of course, and an AMD Radeon RX 6900 XT using ROCm as the compute stack.

The 1080 Ti machine is running **Debian 12** on **Python 3.11**, and the 6900 XT machine is running Arch Linux. The dependencies are:

- CUDA / ROCm and OpenCL (if GPU accelerated compute is desired)
- Torch (pytorch)
- TorchVision
- Pandas
- KaggleHub
- Pillow

The cross-platform python dependencies can be installed with the following pip command:
```
pip install torch torchvision pandas kagglehub pillow
```

Whereas the compute libraries will be more platform-specific, and may require proprietary drivers.

