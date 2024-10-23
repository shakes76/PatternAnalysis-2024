"""
Contains files that mainly handle data tracking from the training
and evaluation loops of the GFNet model.

Benjamin Thatcher
s4784738
"""

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
    eta_min = 1e-6


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
        eta_min
    )

'''
Returns the paths to the ADNI datasets. The path should contain 'train' and
'test' subfolders, each of which should contain 'AD' and 'NC' subfolders
of brain images.
'''
def get_path_to_images():
    #train_path = '/home/groups/comp3710/ADNI/AD_NC/train'
    #test_path = '/home/groups/comp3710/ADNI/AD_NC/test'
    train_path = '../AD_NC/train'
    test_path = '../AD_NC/test'

    return train_path, test_path
