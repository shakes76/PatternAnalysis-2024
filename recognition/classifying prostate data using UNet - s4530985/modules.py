#the source code of the components of your model. Each component must be
#implementated as a class or a function

import torch
import torch.nn as nn



def test_GPU_connection(force_CPU):
    '''
    tests whether pytorch can detect a GPU.
    If no GPU is detected, prints an error message and ends the program.
    Above behaviour overridden by force_CPU, given at runtime

    Parameters:
    force_CPU (Bool): if true, the model will use CPU to make the model. default: false
    
    return:
        none
    '''
    return None

