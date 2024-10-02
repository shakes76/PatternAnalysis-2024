import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import random as rnd

from modules import *
from dataset import *

data = Dataset()
device = data.device

encoder = Encoder().to(device)
decoder = Decoder().to(device)
unet = UNet().to(device)

optimizer = torch.optim.Adam(list(encoder.parameters()) + 
                             list(decoder.parameters()) + 
                             list(unet.parameters()), lr=1e-4)
criterion = nn.CrossEntropyLoss()

loss = []

def train_model(dataloader = data.train_dataloader, epochs=10):
    encoder.train()
    decoder.train()
    unet.train()
    
    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            images = data[0].to(device)
            b_size = images.size(0)
            labels = data[1].to(device)

            t = rnd.randint(0,99)
            beta = torch.linspace(0.001, 0.02, t)
            diffused, noise = forward_diffusion(images, t, beta)
            print(len(diffused))
            optimizer.zero_grad()
            outputs = encoder(diffused[rnd.randint(0,99)])
            print(outputs.shape)
            loss = criterion(outputs, noise)
            loss.backward()
            optimizer.step()

            plt.figure(figsize=(8,8))
            plt.axis("off")
            plt.title("64 Samples of Training Images")
            plt.imshow(np.transpose(vutils.make_grid(outputs.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
            plt.show()

train_model()
