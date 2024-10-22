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

    num_classes = 2

    embed_dim = 512

    depth = 19

    mlp_ratio = 4

    drop_rate = 0.1

    drop_path_rate = 0.1

    weight_decay = 1e-4

    t_max = 8

    eta_min = 1e-6


    return (
        epochs,
        learning_rate,
        patch_size,
        embed_dim,
        num_classes,
        depth,
        mlp_ratio,
        drop_rate,
        drop_path_rate,
        weight_decay,
        t_max,
        eta_min
    )
