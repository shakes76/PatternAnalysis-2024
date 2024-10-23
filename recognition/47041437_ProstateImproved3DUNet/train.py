import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

################### Uncomment to test UNet3D
#from module_unet3D import UNet3D

from module_improvedunet3D import UNet3D
from dataset import augmentation, load_data_3D


class Dice_loss(nn.Module):
    def __init__(self, smooth=0.1):
        super(Dice_loss, self).__init__()
        self.smooth = smooth
   
    
    def label_loss(self, pedictions, targets, smooth=0.1):
        intersection = (pedictions * targets).sum()  
        total = pedictions.sum() + targets.sum()                         
        dice_coeff = (2.0 * intersection + smooth) / (total + smooth)  
        return dice_coeff
    
    # Calculate DSC for each channel, add them up and get the mean
    def forward(self, pedictions, targets, smooth=0.1):    
        
        # Predictions and targets for each label
        prediction_0 = (pedictions.argmax(1) == 0) 
        target_0 = (targets == 0) 

        pediction_1 = (pedictions.argmax(1) == 1) 
        target_1 = (targets == 1) 

        pediction_2 = (pedictions.argmax(1) == 2) 
        target_2 = (targets == 2) 

        pediction_3 = (pedictions.argmax(1) == 3) 
        target_3 = (targets == 3) 

        pediction_4 = (pedictions.argmax(1) == 4) 
        target_4 = (targets == 4) 

        pediction_5 = (pedictions.argmax(1) == 5) 
        target_5 = (targets == 5) 

        # Calculates DSC for each label
        label_0 = self.label_loss(prediction_0, target_0)
        label_1 = self.label_loss(pediction_1, target_1)
        label_2 = self.label_loss(pediction_2, target_2)
        label_3 = self.label_loss(pediction_3, target_3)
        label_4 = self.label_loss(pediction_4, target_4)
        label_5 = self.label_loss(pediction_5, target_5)
        
        # Total DSC averaged over all labels
        dice = (label_0 + label_1 + label_2 + label_3 + label_4 + label_5) / 6.0    
        
        return 1 - dice, {
        'Label 0': label_0,
        'Label 1': label_1,
        'Label 2': label_2,
        'Label 3': label_3,
        'Label 4': label_4,
        'Label 5': label_5,
    }

'''
main() trianing loop. 
Saves model as improved_unet3D.pth
Epoch number: 40-45
Batch size: 1
Loss functions: cross-entropy dice-loss
Optimiser: Adam
'''
def main():
    os.chdir(os.path.dirname(__file__))

    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    # Call model and initialise loss and optimiser parameters
    # model = UNet3D().to(device) -> for 3DUNET original
    model = UNet3D(1,6).to(device)
    augment = augmentation()
    entropy_loss = nn.CrossEntropyLoss().to(device)
    dice_loss = Dice_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Parameters
    epoch = 45
    loss = []
    validation_dsc = {'Label 0': [], 'Label 1': [], 'Label 2': [], 'Label 3': [], 'Label 4': [], 'Label 5': []}
    testing_dsc = {'Label 0': [], 'Label 1': [], 'Label 2': [], 'Label 3': [], 'Label 4': [], 'Label 5': []}

    # Load dataset and split dataset
    dataset = load_data_3D("/home/groups/comp3710/HipMRI_Study_open/semantic_MRs/*", "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only/*")
    train, validate, test = torch.utils.data.random_split(dataset, [181, 15, 15])

    # Set your desired batch size here
    batch_size = 1  
    train = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    validate = DataLoader(validate, batch_size=batch_size, shuffle=False, pin_memory=True)
    test = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True)

    for i in range(epoch) :
        # Training
        model.train()
        for image, label in train :
            image = image.squeeze(0)
            label = label.squeeze(0)
            image, label = augment.augment(image, label)
            # !!!Use repeat or expand to transfer batchsize value if batchsize > 1
            image = image.unsqueeze(0)
            image = image.float().to(device)
            label = label.long().to(device)
            optimizer.zero_grad()
            pred = model(image)
            loss = entropy_loss(pred, label)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            validation_dsc_epoch = {k: 0 for k in validation_dsc.keys()}
            for image, label in validate:
                label = label.squeeze(0)
                image = image.float().to(device)
                label = label.long().to(device)
                pred = model(image)
                dice, dice_score = dice_loss(pred, label)
                for labels in validation_dsc_epoch:
                    validation_dsc_epoch[labels] += dice_score[labels]
            
            # Store average DSC per label for this epoch
            for labels in validation_dsc:
                validation_dsc[labels].append(validation_dsc_epoch[labels].item() / len(validate))

        print('\nOne Epoch Finished', flush = True)
        print(f"Validation DSC: \n", flush = True)
        for labels, score in dice_score.items():
            print(f"{labels}: {score:>8f}", flush = True)
        
        torch.save(model.state_dict(), 'improved_UNet3D.pth')

        # Testing
        model.eval()
        with torch.no_grad():
            testing_dsc_epoch = {k: 0 for k in testing_dsc.keys()}
            for image, label in test:
                label = label.squeeze(0)
                image = image.float().to(device)
                label = label.long().to(device)
                pred = model(image)
                dice, dice_score = dice_loss(pred, label)
                for labels in testing_dsc_epoch:
                    testing_dsc_epoch[labels] += dice_score[labels]

            for labels in testing_dsc:
                testing_dsc[labels].append(testing_dsc_epoch[labels].item() / len(test))

        print(f"Testing DSC: \n", flush = True)
        for labels, score in dice_score.items():
            print(f"{labels}: {score:>8f}", flush = True)


    epochs = range(1, epoch + 1)
    for label in testing_dsc:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, validation_dsc[label], label='Validation')
        plt.plot(epochs, testing_dsc[label], label='Test')
        plt.title(f'Dice Similarity Coefficient for {label}')
        plt.xlabel('Epoch')
        plt.ylabel('DSC')
        plt.legend()
        plt.savefig(f'dsc_plot_label_{label}.png')

if __name__ == "__main__":
    main()