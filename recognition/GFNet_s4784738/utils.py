"""
Contains the model parameters and paths to the image files to 
be used for training, testing, and predicting.

Benjamin Thatcher
s4784738
"""

import random
import os

'''
All necessary parameters for training the model
'''
def get_parameters():
    epochs = 30
    learning_rate = 1e-4
    patch_size = (16, 16)
    embed_dim = 512
    depth = 19
    mlp_ratio = 4
    drop_rate = 0.1
    drop_path_rate = 0.1
    weight_decay = 1e-2
    t_max = 6

    return (
        epochs,
        learning_rate,
        patch_size,
        embed_dim,
        depth,
        mlp_ratio,
        drop_rate,
        drop_path_rate,
        weight_decay,
        t_max,
    )

def get_path_to_images():
    '''
    Returns the paths to the ADNI datasets. The training and testing paths each 
    should contain 'AD' and 'NC' subfolders of brain images.
    '''
    train_path = '/home/groups/comp3710/ADNI/AD_NC/train'
    test_path = '/home/groups/comp3710/ADNI/AD_NC/test'

    return train_path, test_path

def get_prediction_image():
    '''
    Returns the path to an image whose classification will be predicted in predict.py.
    A random image from the testing set will be selected if a set img_path is not set.
    '''
    # A set image path to be returned
    img_path = None

    # Select a random image if necessary
    if not img_path:
        _, test_path = get_path_to_images()
        # Pick AD or NC image with 50-50 probability
        real_class = random.randint(0, 1)

        classification = 'AD' if real_class == 1 else 'NC'
        img_path = os.path.join(test_path, classification)

        # Pick an image at random
        imgs = [image for image in os.listdir(img_path)]
        chosen_img = random.choice(imgs)
        img_path = os.path.join(img_path, chosen_img)

        print(f'The randomly selected image (found at {img_path}) is truly {classification}')

    return img_path