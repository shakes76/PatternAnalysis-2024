#the source code for training, validating, testing and saving your model.
# The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”.
# Make sure to plot the losses and metrics during training

#foreign libraries
import argparse

#local libraries
import modules
import dataset


def validate_UNet():
    '''
    function for validating the model.
    should only ever be called after the model has been trained on train/test data
    validation data must be split off apart from train/test data
    Parameters:
    
    
    return:
        validation metric
    '''
    return None

def train_UNet():
    '''
    function for training the model.
    Parameters:


    return:
        updated model?
    '''
    return None

def test_UNet():
    '''
    function for testing the model.
    Parameters:


    return:
        updated model?
    '''
    return None

def Save_UNet():
    '''
    function for saving the model????????????
    Parameters:


    return:
        huh?????????
    '''
    #torch.save(model.state_dict(), "Unet.pth")
    return None



def create_Unet(train, test, size, force_CPU = False):
    '''
    ideally, this should be the only function a person wishing to use this package will need to call
    Parameters:
        train (Dataset_3d): the segment of the data to be trained on
        test (Dataset_3d): the segment of the data to be tested on
        size (int): the N*N*N dimension of the train & test data
        force_CPU (bool): if true, allow the program to run on CPU, otherwise will abort if no GPU detected
        
    return:
        The trained Unet model
    '''
    device = modules.test_GPU_connection(force_CPU)
    #load
    epochs = 10
    batch_size = 4
    dim = size
    


    
    print("test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("File_path", help="file path to the dataset to be ??????? modelled on?????")
    parser.add_argument("-fc","--force_CPU", help="set if you wish to allow usage of CPU, if GPU not available", action="store_true", default=False)
    #parser.add_argument("dimensionality", help="specify wether the data is a 2d or 2d segment", type=int, default=3)
    parser.add_argument("size", help="the N*N*N dimensions of the images in the dataset", type=int)
    args = parser.parse_args()

    
    create_Unet(args.File_path, args.size, args.force_CPU)