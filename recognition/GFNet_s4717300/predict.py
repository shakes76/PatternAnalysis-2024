'''
@file   predict.py
@brief  Script used to perform inference on a given model
@author  Benjamin Jorgensen - s4717300
@date   18/10/2024
'''
from dataset import GFNetDataloader
from modules import GFNet
from utils import Environment

import time
import torch
import argparse
from PIL import Image
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


def evaluate_model(model, test_set, criterion, env, device, estimate=True):
    """
    Perform Evaluation of the model based on some test set. Prints out accuracy
    and convolution matrix.

    @params model: The model to evaluate
    @params test_set: Dataloader containing the data to evaluate against
    @params env: Environment containing metadata about the model, hyperparameter and loss values
    @params device: device to run evaluation on
    @params estimate: Save results to test accuracy estimate array or test accuracy array

    @returns accuracy, average_loss
    """
    print("==Testing====================")
    start_eval = time.time() #time generation
    model.eval() 

    all_preds = []  # To store all predictions
    all_labels = []  # To store true labels

    with torch.no_grad():
        correct = 0
        total = 0
        avg_loss = 0 
        count = 0
        for images, labels in test_set:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            count += 1
            correct += (predicted == labels).sum().item()
            avg_loss += criterion(outputs, labels) 

            # Collect predictions and true labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        if estimate:
            env.estimated_test_losses.append(float(avg_loss / count))
        else:
            env.test_losses.append(float(avg_loss / count))

        accuracy = (100 * correct / total)
        if estimate:
            env.estimated_test_accuracy.append(accuracy)
        else:
            env.test_accuracy.append(accuracy)
            # Compute the confusion matrix
            conf_matrix = confusion_matrix(all_labels, all_preds)
            print('Confusion Matrix:\n', conf_matrix)

        print('Test Accuracy: {:.2f} % | Average Loss: {:.4f}'.format(accuracy, avg_loss / count)) 

    end = time.time()
    elapsed = end - start_eval
    print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    model.train()
    print("=============================")
    return accuracy, (avg_loss / count)

if __name__ == '__main__':
    # Setting up CUDA
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    parser = argparse.ArgumentParser(description='Predicting a batch of images or a single image')
    # Non-optional positional argument
    parser.add_argument('model', type=str, help='Path of model to be to use for prediction')
    parser.add_argument('path', type=str, help='Path of image to be to use for inference or directory for evaluation')
    parser.add_argument('-e', '--evaluation', action='store_true', help='Evaluate an entire dataset and output resutls')
    parser.add_argument('-b', '--batch_size', type=int, help='Evaluate an entire dataset and output resutls')

    args = parser.parse_args()

    if not args.path:
        print('Error: Must provide a path for image inference or evaluation')
        exit(1)

    if not args.model:
        print('Error: Must provide a model to use for inference or evaluation')
        exit(1)

    if args.batch_size and not args.evaluation:
        print('Error: Batch size must be used in only in evaluation mode (-e)')

    # If we want to evaluate
    # Load an environment, (not as flexible as in train.py)
    basic_env = Environment()
    batch_size = args.batch_size if args.batch_size else 64
    loader = GFNetDataloader(batch_size)

    if args.evaluation:
        # Load the evaluation batch
        loader.load(args.path)
        train, test, val = loader.get_data()
        meta = loader.get_meta()

        # Image Info
        channels = meta['channels']
        num_classes = meta['n_classes']
        image_size = meta['img_size']
        img_shape = (channels, image_size, image_size)

        if not train or not test:
            print("Problem loading data, please check dataset is commpatable \
                    with dataloader including all hyprparameters")
            exit(1)
    else:
        # Single image inference
        image = Image.open(args.path)
        image_size = min(image.size)
        image_tensor = loader.transform_val(image, image_size)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)
        num_classes = basic_env.n_classes
        channels = 1

    # Handle errors when loading the model
    try:
        model = GFNet(img_size=image_size,
                         patch_size=basic_env.patch_size,
                         in_chans=channels,
                         num_classes=num_classes,
                         embed_dim=basic_env.embed_dim,
                         depth=basic_env.depth,
                         ff_ratio=basic_env.ff_ratio,
                         dropout=basic_env.dropout,
                         drop_path_rate=basic_env.drop_path)
        model.to(device)
        model.load_state_dict(torch.load(args.model, weights_only=False))
    except Exception as e:
        print(e)
        print('==================================================')
        print('Erorr: If you\'re seeing this message, then your model hyperparameters \
                are different to those in Utils.py. Please these values and then re-reun this command')
        exit(1)

    # Perform evaluation
    if args.evaluation:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
        evaluate_model(model, val, criterion, basic_env, device, estimate=False)
        exit(0)

    # Doing inference on a single Image
    output = model(image_tensor)
    probabilities = F.softmax(output, dim=1)
    max_index = torch.argmax(probabilities[0])
    print('====Inference===========================================')
    print('Performing Inference on {}'.format(args.path))
    print('Predicted class {} with {}% liklihood.'.format(max_index, round(float(probabilities[0][max_index]*100), 2)))


