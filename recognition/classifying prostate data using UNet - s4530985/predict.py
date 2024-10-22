#showing example usage of your trained model. Print out any results and / or provide
# visualisations where applicable

##I think the "example" being reffered to is the intended dataset?

import train

########LOOK AT WHAT'S ALREADY ON THE GITHUB AS A GUIDE!!

if __name__ == "__main__":
    #2d data
    train.main("/home/groups/comp3710/HipMRI_Study_open/keras_slices_data")

    #3d data
    train.main("/home/groups/comp3710/HipMRI_Study_open")

    #3d data, improved Unet