"""
Contains files that mainly handle data tracking from the training
and evaluation loops of the GFNet model.

Benjamin Thatcher
s4784738
"""

import random
import os

'''
All necessary parameters for training the model
'''
def get_parameters():
    epochs = 6
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

'''
Returns the paths to the ADNI datasets. The training and testing paths each 
should contain 'AD' and 'NC' subfolders of brain images.
'''
def get_path_to_images():
    #train_path = '/home/groups/comp3710/ADNI/AD_NC/train'
    #test_path = '/home/groups/comp3710/ADNI/AD_NC/test'
    train_path = '../AD_NC/train'
    test_path = '../AD_NC/test'

    return train_path, test_path

'''
Returns the path to an image whose classification will be predicted in predict.py.
'''
def get_prediction_image():
    # A set image path to be returned
    img_path = None

    # If there is no set image path, an image will be selected at random from the test_path
    # returned from get_path_to_images
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