'''
@file   predict.py
@brief  Script used to perform inference on a given model
@author  Benjamin Jorgensen - s4717300
@date   18/10/2024
'''
import time
import torch
from train import Environment

# Setting up CUDA
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)


def evaluate_model(model, test_set, criterion, env: Environment, device, estimate=True):
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
        print('Test Accuracy: {:.2f} % | Average Loss: {:.4f}'.format(accuracy, avg_loss / count)) 

    end = time.time()
    elapsed = end - start_eval
    print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    model.train()
    print("=============================")
    return accuracy, (avg_loss / count)
