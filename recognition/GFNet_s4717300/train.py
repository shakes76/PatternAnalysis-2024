'''
@file   train.py
@brief  Script used to train the GFNet
@author  Benjamin Jorgensen - s4717300
@date   18/10/2024
'''
from dataset import GFNetDataloader
from modules import GFNet
import predict
import torch.optim.adamw
import torch
import time
from torch.optim.lr_scheduler import OneCycleLR
import csv
import argparse
import os

# Setting up CUDA
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print(device)

class Environment():
    """
    Container for hyperparameter and other miscellaneous
    """
    def __init__(self) -> None:
        # Hyperparameter
        self.learning_rate  = 0.001 
        self.weight_decay   = 0.001
        self.dropout        = 0.0
        self.drop_path      = 0.1       # Probability of dropping an entire network path from an iteration
        self.batch_size     = 32 
        self.patch_size     = 16        # Size of the chunks to split up the image in pixels
        self.embed_dim      = 783 
        self.depth          = 8         # Number of global filter layers to use in the network
        self.ff_ratio       = 3         # Ratio of input to hidden layer size in the classification feed forward network
        self.epochs         = 100

        # Model Metadata
        self.dataset_path   = None      # Path of training and testing data
        self.save_check     = False     # True if use wants to save checkpoints every 10 epochs
        self.model_path     = None      # Path of a model checkpoint to be loaded
        self.tag            = "GFNet"   # Name of the model, will be appended to all output

        # Data collection
        self.training_losses = []
        self.estimated_test_losses = []
        self.test_losses = []
        self.estimated_test_accuracy = []
        self.test_accuracy = []


def setup_environment() -> Environment:
    """
    Creates an environment based on default hyperparameter and parsed arguments

    @returns: Environment
    """
    env = Environment()

    # Define custom arguments
    parser = argparse.ArgumentParser(description='Custom parameters for training the model')
    parser.add_argument('-c', '--checkpoint', type=str, help='Model weights to load')
    parser.add_argument('-s', '--save_checkpoint', type=str, help='Optionally save checkpoints as every 10 epochs')
    parser.add_argument('-t', '--tag', type=str, help='Tag to label the models')

    # hyperparameter arguments
    parser.add_argument('-b', '--batch_size', type=str, help='tag to label the models')
    parser.add_argument('-d', '--depth', type=str, help='tag to label the models')
    parser.add_argument('-l', '--learning_rate', type=str, help='tag to label the models')
    parser.add_argument('-e', '--epochs', type=str, help='Number of epochs to run the model for')

    # Non-optional positional argument
    parser.add_argument('dataset_path', type=str, help='directory location of dataset')

    # Parse the arguments
    args = parser.parse_args()

    # Updating environment with parsed arguments
    if not args.dataset_path:
        print('Error: No path given for training and test data. See useage for details')
        exit(1)

    if args.checkpoint: env.model_path = args.checkpoint
    if args.save_checkpoint: env.save_check = args.save_check
    if args.tag: env.tag = args.tag

    if args.batch_size: env.batch_size = args.tag
    if args.depth: env.depth = args.depth
    if args.learning_rate: env.learning_rate = args.learning_rate
    if args.epochs: env.epochs = args.epochs

    # Create the directory if it doesn't exist
    os.makedirs(env.tag, exist_ok=True)
    return env


def train_model(model, env, train, test, validation):
    """
    The main training loop of the model. Uses all the parameters in the env
    variable. Returns nothing but saves all output to the tag directory.

    @params model: The model to be trained, can be blank or loaded from a checkpoint
    @params env: Contains all information about the environment
    @params train: Dataloader containing the training images
    @params test: Dataloader containing the FULL set of images to be judged for accuracy
    @params validation: Dataloader containing a SUBSET of images to be judged for accuracy
    """
    # Record time of training start
    print("==Training====================")
    start = time.time() #time generation

    # Begin main training loop
    for epoch in range(env.epochs):
        for i, (images, labels) in enumerate(train, 0):
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            train_loss = criterion(output, labels)

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            env.training_losses.append(train_loss.item())
        
            # Print out progress
            if i % 10 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}" .format(epoch+1, epochs, i+1, len(training), train_loss.item())) #type: ignore

        if epoch % 10 == 0:
            torch.save(model.state_dict(), '{}/Checkpoint-epoch{}-{}.pth'.format(env.tag, epoch, env.tag))
            output_results(env)

        scheduler.step() 

    torch.save(model.state_dict(), '{}/{}.pth'.format(env.tag))
    output_results(env)

    # End timer
    print("==Finished training====================")
    end = time.time() #time generation
    elapsed = end - start
    print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    model.train()


def output_results(env: Environment):
    """
    Saves results and losses to a csv file

    @param env: Environment containing metadata such as hyperparameter and loss logs
    """
    with open('{}/losses.csv'.format(env.tag), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(env.training_losses)
    with open('{}/test_losses.csv'.format(env.tag), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(env.test_losses)
    with open('{}/test_accuracy.csv'.format(env.tag), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(env.test_accuracy)
    with open('{}/est-test_losses.csv'.format(env.tag), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(env.estimated_test_losses)
    with open('{}/est-test_accuracy.csv'.format(env.tag), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(env.estimated_test_accuracy)


if __name__ == '__main__':
    env = setup_environment()

    # Load data
    gfDataloader = GFNetDataloader(env.batch_size)
    gfDataloader.load(env.dataset_path)

    training, test, validation = gfDataloader.get_data()
    meta = gfDataloader.get_meta()

    # Image Info
    channels = meta['channels']
    num_classes = meta['n_classes']
    image_size = meta['img_size']
    img_shape = (channels, image_size, image_size)

    if not training or not test:
        print("Problem loading data, please check dataset is commpatable with dataloader including all hyprparameters")
        exit(1)


    model = GFNet(img_size=image_size,
                     patch_size=env.patch_size,
                     in_chans=channels,
                     num_classes=num_classes,
                     embed_dim=env.embed_dim,
                     depth=env.depth,
                     ff_ratio=env.ff_ratio,
                     dropout=env.dropout,
                     drop_path_rate=env.drop_path)
    model.to(device)
    model.train()
    if env.model_path:
        model.load_state_dict(torch.load(env.model_path, weights_only=False))

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=env.learning_rate, weight_decay=env.weight_decay) #type: ignore
    scheduler = OneCycleLR(optimizer,max_lr=env.learning_rate, steps_per_epoch=len(training), epochs=env.epochs)

    train_model(model, env, training, test, validation)
    predict.evaluate_model(model, validation, criterion, env, device, False)

