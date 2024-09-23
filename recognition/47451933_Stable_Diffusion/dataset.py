import torch
import torchvision
import torch.utils.data.dataloader as dataloader
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

#=======================================================
# key variables
#=======================================================
IMAGES = 0
LABELS = 1
#=======================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
show_images = True
#=======================================================

#paths to datasets
adni_data_path_train = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'train'))
adni_data_path_test = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'ADNI', 'AD_NC', 'test'))

transforms = transforms.Compose([
    transforms.Grayscale(), #grascale images 
    transforms.Resize(64), #the next two lines decrease the resolution to 64x64
    transforms.CenterCrop(64),
    transforms.ToTensor(), #turn the datat into a tensor if its not already
    transforms.Normalize(0.5,0.5)]) #normilze the data 0.5 beacuse values between 0-1 so 0.5 is just good general value

#create datasets from image folder
train_dataset = torchvision.datasets.ImageFolder(root=adni_data_path_train, transform=transforms)
test_dataset = torchvision.datasets.ImageFolder(root=adni_data_path_test, transform=transforms)

print(train_dataset)
print(test_dataset)

#create data loaders
print("Creating DataLoaders...⏳") #setup data loadeers for training to make thins easier with batching
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print("DataLoaders Created ✅")

#get sample of data
images = next(iter(train_dataloader))
print(len(images[IMAGES]), len(images[LABELS]))
images = images[IMAGES]

#visualise the sample
print("Showing Data Sample...⏳")
print(images.shape)
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("64 Samples of Training Images")
plt.imshow(np.transpose(vutils.make_grid(images.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
if show_images: plt.show()
print("Data Sample shown ✅")