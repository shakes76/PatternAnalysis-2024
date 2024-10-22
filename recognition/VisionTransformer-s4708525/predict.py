from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torch.nn as nn
import time
import torch.optim as optim
from modules import *
from train import *
from dataset import *

if __name__ == "__main__":
    """
    Main functin to do the prediction of the existing model
    """

    path = 'cvt_model.pth'

    # initialize model
    model = CvT()

    # load model
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.eval()

    # path image to be predicted
    image_path_server = f'{test_path_server}/AD/1254307_109.jpeg'
    input_image = preprocess_image(image_path_server) 
    input_image = input_image.to(device) 

    with torch.no_grad():  
        output = model(input_image)  
        _, predicted_class = torch.max(output, 1)  
        predicted_class = predicted_class.item()  

    # change it based on the label of the images to be predicted
    is_AD = True
    is_NC = False

    ground_truth = "AD" if is_AD else "NC"

    print('Labels corresponding to the images (0 for NC, 1 for AD)')
    print('Ground truth class:', ground_truth)
    print(f'Predicted class: {predicted_class}')

    if is_AD:
        results = evaluate_prediction(predicted_class, 1)
    
    if is_NC:
        results = evaluate_prediction(predicted_class, 0)
    
    print(results)




