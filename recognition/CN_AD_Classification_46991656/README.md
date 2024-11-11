# Alzheimer’s Disease Classification Using Vision Transformer (46991656)

## Description
This project aims to identify Alzheimer’s disease (AD) using MRI brain images from the Alzheimer’s Disease Neuroimaging Initiative (ADNI). Our goal is to distinguish between healthy individuals and those diagnosed with Alzheimer's disease. To achieve this, we use the Vision Transformer, a modern machine learning model known for its ability to analyze images effectively. For this model, we classify images into two categories: health individuals (cognitive normal) and those diagnosed with Alzheimer's. We hope to achieve an accuracy of 80%, which can aid in early diagnosis and better treatment options for patients.
<br>
<br>

## How It Works
The Vision Transformer model processes brain images by dividing them into smaller sections, called patches. Each patch is then transformed into a higher-dimensional space through an embedding. We also use a class token to represent the overall image, along with positional information for each patch.

The model uses a series of layers that work together to recognize patterns and features in the images, which helps it determine whether an image shows a healthy brain or one affected by Alzheimer’s. Finally, the model outputs a prediction based on what it has learned from the images.

An image showing the breakdown of this is underneath. (Source: https://paperswithcode.com/method/vision-transformer)
<br>
<br>

![alt text](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png)
<br>
<br>

### TransformerBlock Class Summary
The `TransformerBlock` class implements a single layer of the Vision Transformer model, consisting of a multi-head self-attention mechanism and a feedforward neural network (MLP) with GELU activations. The self-attention layer allows each patch to attend to all other patches, capturing spatial relationships within the image. Each attention and MLP output is processed with residual connections and layer normalisation, which helps stabilise training and improves the network's ability to learn complex representations. Stacking multiple `TransformerBlock` layers enables the model to refine the patch interactions and learn more detailed spatial dependencies.
<br>

### SimpleViT Class Summary
The `SimpleViT` class defines the overall Vision Transformer model, which classifies images by processing them as sequences of patches. The model splits an input image into 16x16 patches, embeds these patches into a higher-dimensional space, and adds positional information to retain the spatial context. A learnable class token is prepended to the patch sequence, and the combined embeddings are passed through multiple `TransformerBlock` layers. After this, only the class token's representation is used for final classification through a small MLP head, which outputs class probabilities. This structure allows the model to learn global representations across patches, making it effective for tasks like image classification.
<br>

### Training Process Summary
The training process for the Vision Transformer (ViT) model involves several steps to optimise the model for accurate image classification. Initially, training and test data loaders are set up using a custom dataset loader, and the ViT model is initialised with specific parameters for image patching and transformation. A weighted cross-entropy loss function and the Adam optimiser with weight decay are used to minimise classification errors, while a learning rate scheduler reduces the learning rate adaptively based on training loss. During each epoch, the model processes batches of images, preforming a forward pass to generate predictions, computes the loss, and adjust weights through backpropagation to improve performance. After each epoch, metrics like average loss and accuracy are recorded and printed for monitoring progress. Following training, the model is evaluated on the test set, with test accuracy and loss logged to assess its generalisation ability.

Finally, the training and testing metrics, such as loss and accuracy curves, are plotted and saved as a visual representation of the model's progress. The trained model's weights are saved to a file, which enables reloading the model for future inference tasks without retraining. This training script, which relies on `dataset.py` and `modules.py`, provides a comprehensive setup for training, validating, testing, and saving the ViT model effectively.
<br>
<br>

## Dependencies
To run this code, ensure you have `Python >= 3.1`, `PyTorch >= 1.9.0`, `torchvision >= 0.10.0`, and `matplotlib >= 3.4.3` installed. To do this, you can use the `pip install` or `conda install` commands.

Ensure that the four files, `train.py`, `modules.py`, `train.py`, and `predict.py`, are all in the same working directory. The `train.py` file has imports from both `dataset.py` and `modules.py`, to access the dataloaders and the vision transformer model. Make sure to update the filepath of the `train_dataset` and `test_dataset` to where the image files are located.

Run `train.py` to train the model, and save it inside of the current directory. Once finished, run the `predict.py` file to get the accuracy of the model, ensuring that the name of the saved model is kept consistent between both files. This is set in the code as `vit_adni_model.pth`.

Note that this project involves some randomised processes (such as random weight initialisation and data shuffling) which may result in moderately different outputs or model performance on each run. Therefore, the behaviour of the model, including accuracy and metrics, may vary.
<br>
<br>

## Example Usage
The only inputs to the model are the training images and testing images. These are located in the filepaths set up in `dataset.py`, so ensure that you change this based on where your images are located locally.

After running the `train.py` file, it will print out statements of the loss and accuracy corresponding to each epoch, and then run a test on it. Finally, it will save the vision transformer model in the same directory.
<br>
<br>

```
Running main loop now.
Beginning training now.
Epoch [1/20], Loss: 0.6679, Accuracy: 56.36%
Epoch [2/20], Loss: 0.5900, Accuracy: 67.40%
Epoch [3/20], Loss: 0.5570, Accuracy: 70.62%
Epoch [4/20], Loss: 0.5383, Accuracy: 71.96%
Epoch [5/20], Loss: 0.5198, Accuracy: 73.52%
Epoch [6/20], Loss: 0.5083, Accuracy: 74.52%
Epoch [7/20], Loss: 0.4947, Accuracy: 75.62%
Epoch [8/20], Loss: 0.4832, Accuracy: 76.00%
Epoch [9/20], Loss: 0.4724, Accuracy: 77.14%
Epoch [10/20], Loss: 0.4593, Accuracy: 77.79%
Epoch [11/20], Loss: 0.4519, Accuracy: 78.23%
Epoch [12/20], Loss: 0.4404, Accuracy: 78.66%
Epoch [13/20], Loss: 0.4291, Accuracy: 79.59%
Epoch [14/20], Loss: 0.4174, Accuracy: 80.51%
Epoch [15/20], Loss: 0.4015, Accuracy: 81.40%
Epoch [16/20], Loss: 0.3910, Accuracy: 81.71%
Epoch [17/20], Loss: 0.3852, Accuracy: 82.31%
Epoch [18/20], Loss: 0.3731, Accuracy: 83.03%
Epoch [19/20], Loss: 0.3654, Accuracy: 83.50%
Epoch [20/20], Loss: 0.3580, Accuracy: 83.75%
Beginning testing now.
Test Loss: 0.8216, Test Accuracy: 63.30%
Model saved with Test Accuracy: 63.30%
```
<br>

While the model did excellent in the training phase and achieved a percentage accuracy of 83.75% after 20 epochs, it did poorly on the actual test set. This is most likely to the model overfitting on the training set, and not being able to generalise well enough overall. In future, dropout or L2 regularisation could be used more to improve this aspect.

<br>

An output from the training is a measure of the loss and accuracy, including a dotted line for comparison with the accuracy on the testing set.
<br>
<br>

![alt text](graphs/training_metrics_plot.png)
<br>
<br>

After running the `train.py` file, you should see an object in the local directory called `vit_adni_model.pth`. This is a saved version of the trained vision transformer model, and is used in `predict.py` to run a test on it. A sample output is below.
<br>
<br>

```
Starting predictions...
Accuracy of the model: 63.30%
```
<br>

## Data Pre-processing

In this project, the training data went under many pre-processing steps to enhance the model performance and generalisation.

All the images were resized to 224x224 pixels to maintain a consistent input size for the Vision Transformer model. Data augmentation was applied as well, introducing random horizontal flips, random rotations, and colour disfigurations. The images were also normalised using the mean and standard deviation of the ImageNet dataset, to ensure that pixel values are scaled to a standard range and improve convergence during training.

The testing set only used resizing and normalisation, since the data augmentation could lead to ambiguous evaluation metrics. As two separate directories were provided for training and testing, the split between them was quite obvious, but in cases where all the data is combined, roughly 80% of the data should be used for training and the other 20% for testing. This helps mitigate issues like overfitting, by confirming that the model's performance is validated on a separate dataset.
<br>
<br>

## Future Recommendations
One of the main adjustments to the parameters was reducing the patch size from 32 down to 16, which greatly increased the level of accuracy in the training stage. While the model did not achieve the test target of 80%, there are a few things that can be done to help improve the accuracy. For instance, experimenting with different hyperparameters, such as adjusting the learning rate and batch size, may lead to better model performance. Additionally, exploring alternative model architectures could provide a more robust feature extraction capability. Another approach could be to increase the size of the training dataset by incorporating more diverse samples, which may help the model generalise better to unseen data. This could help improve the testing accuracy greatly. Lastly, implementing techniques like cross-validation could provide a more accurate assessment of the model’s performance and guide further improvements.
<br>
<br>

## Conclusion
This project demonstrates the potential of Vision Transformer models for Alzheimer's disease classification using MRI brain images. By using ViT's approach of splitting images into patches and learning spatial relationships, the model achieved promising results in identifying Alzheimer's disease in the training phase, reaching an accuracy of 83.75%. However, the model struggled to generalise to the test set, with a final test accuracy of 63.30%, indicating overfitting.

The outcomes highlight both the strengths and limitations of using ViT for medical image classification. While the model was effective in learning patterns during training, additional regularisation techniques such as droput or L2 regularisation may be needed to improve its generalisation capability. Future efforts could focus on hyperparameter tuning, and testing alternative model architectures to enhance classification performance.

Overall, this project shows the viability of Vision Transformers in early diagnosis efforts for Alzheimer's disease, while also providing insights into the challenges of model generalisation in clinial applications. Continued experimentation and optimisation could ultimately help improve early detection, offering a valuable tool for clinicians and researchers working to advance Alzheimer's treatment and care.

