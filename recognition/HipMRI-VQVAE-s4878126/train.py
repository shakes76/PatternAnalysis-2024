from modules import Encoder, Decoder, Quantise, Model
from utils import parameters, dimensions
import dataset
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

print(f"CUDA available? {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"VQVAE is being trained on {device}.\n")
else:
    print(f"GPU not available, VQVAE is switching to {device}.\n")

train_set = dataset.GetTrainSet()
train_loader = DataLoader(train_set, batch_size=parameters["batch"], shuffle=True)

print(f"Train.py: Shape of first training dataset image {train_set[0].shape}")

encoder = Encoder(inputDim=dimensions["input"], hiddenDim=dimensions["hidden"], outputDim=dimensions["latent"])
quantise = Quantise(nEmbeddings=dimensions["embeddings"], embed_dim=dimensions["latent"])
decoder = Decoder(inputDim=dimensions["latent"], hiddenDim=dimensions["hidden"], outputDim=dimensions["output"])

vqvae = Model(Encoder=encoder, Quantise=quantise, Decoder=decoder).to(device=device)

mse_loss = nn.MSELoss()
optimiser = Adam(vqvae.parameters(), lr=parameters["lr"])
# optimiser = SGD(vqvae.parameters(), lr=parameters["lr"], momentum=0.9)

print("Training the Vector Quantised Variational Autoencoder...")
vqvae.train()

epoch_values = np.linspace(0, 50, 50)

reconloss_values = []
cbloss_values = []
cmloss_values = []
pxloss_values =  []
total_loss = []

for epoch in range(parameters["epochs"]):
    running_loss = 0.0
    rc_loss = 0.0
    cd_loss = 0.0
    cm_loss = 0.0
    p = 0.0

    for i, image in enumerate(train_loader):
        image = image.to(device)

        optimiser.zero_grad()
        
        imageRec, commitment_loss, codebook_loss, perplexity = vqvae(image)
        reconstr_loss = mse_loss(imageRec, image)

        loss = reconstr_loss + commitment_loss * dimensions["commitment_beta"] + codebook_loss
        running_loss += loss.item()
        rc_loss = reconstr_loss.item()
        cd_loss = codebook_loss.item()
        cm_loss = commitment_loss.item()
        p = perplexity.item()

        loss.backward()
        optimiser.step()

        if i % 64 == 0:
            print(f"Loss for {epoch + 1},{i + 1} = {loss.item()}\nreconstruction loss = {reconstr_loss.item()}, perplexity: {perplexity.item()}, commitment loss: {commitment_loss.item()}, codebook loss: {codebook_loss.item()}")
    print(f"Epoch {epoch + 1} loss: {running_loss}. ")
    reconloss_values.append(rc_loss)
    cbloss_values.append(cd_loss)
    cmloss_values.append(cm_loss)
    pxloss_values.append(p)
    total_loss.append(running_loss)


def recordQuantiseLoss(epochs, epoch_values, reconloss_values, cbloss_values, cmloss_values):
    plt.clf()
    plt.title(f"Decomposing Training Loss for the VQVAE over {epochs} Epochs.")
    plt.plot(epoch_values, reconloss_values, label="Reconstruction Loss", color='blue')
    plt.plot(epoch_values, cbloss_values, label="Codebook Loss", color='orange')
    plt.plot(epoch_values, cmloss_values, label="Commitment Loss", color='green')
    plt.legend(loc='upper right', title='Loss values')
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.savefig('VQVAE_Loss_Functions.png')

def recordLoss(epoch_values, loss_values, y_axis_label, title, fig_name):
    plt.clf()
    plt.title(f"{title}")
    plt.plot(epoch_values, loss_values)
    plt.xlabel("Epochs")
    plt.ylabel(f"{y_axis_label}")
    plt.savefig(f'{fig_name}')

recordQuantiseLoss(epochs=parameters["epochs"], epoch_values=epoch_values, reconloss_values=reconloss_values, cbloss_values=cbloss_values, cmloss_values=cmloss_values)
recordLoss(epoch_values=epoch_values, loss_values=pxloss_values, y_axis_label="Perplexity", title="VQVAE Perplexity over Epochs", fig_name="VQVAE_Perplexity.png")
recordLoss(epoch_values=epoch_values, loss_values=total_loss, y_axis_label="Total Loss", title="VQVAE Total Loss over Epochs", fig_name="VQVAE_TotalLoss.png")

torch.save(vqvae.state_dict(), 'hipmri_vqvae.pth')
