import torch
import torch.nn as nn
import torch.nn.functional as F

"""
REFERENCES:
Kang, J. (2024, Feburary 15). Pytorch-VAE-tutorial. Retrieved 31st August 2024 from 
    https://github.com/Jackson-Kang/Pytorch-VAE-tutorial
Rosinality. (2021, January 23). VQ-VAE-2-PYTORCH. Retrieved 2nd September 2024 from https://github.com/rosinality/vq-vae-2-pytorch
Yadav, S. (2019, September 1). Understanding Vector Quantized Variational Autoencoders (VQ-VAE) [Blog]. Medium.
    https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a 
"""

"""Encodes the images into latent space,
    with lower granularity, by parameterising the a categorical
    posterior distribution over the latent space.
    In a normal VAE, the posterior and prior assume a 
    continuous normal distribution instead of categorial (Yadav, 2019)."""
class Encoder(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim,
                 kernels=(4, 4, 3, 1), stride=2):
        super().__init__()

        k1, k2, k3, k4 = kernels

        self.stridedConv1 = nn.Conv2d(inputDim, hiddenDim, kernel_size=k1, stride=stride, padding=1)
        self.stridedConv2 = nn.Conv2d(hiddenDim, hiddenDim, kernel_size=k2, stride=stride, padding=1)
        
        self.residualConv1 = nn.Conv2d(hiddenDim, hiddenDim, kernel_size=k3, padding=1)
        self.residualConv2 = nn.Conv2d(hiddenDim, hiddenDim, kernel_size=k4, padding=0)

        self.proj = nn.Conv2d(hiddenDim, out_channels=outputDim, kernel_size=1)

    def forward(self, inp):
        inp = self.stridedConv1(inp)
        inp = self.stridedConv2(inp)

        inp = nn.ReLU()(inp)
        out = self.residualConv1(inp)
        out += inp

        inp = nn.ReLU()(inp)
        out = self.residualConv2(inp)
        out += inp

        out = self.proj(out)

        return out

"""
Quantisation refers to comparing the encoder output against a previously stored dictionary of embeddings. 
The distance between each encoded vector and embedding is calculated, and the encoded vector is replaced with its nearest embedding.
this quantised output is passed to the decoder.
"""
class Quantise(nn.Module):
    def __init__(self, nEmbeddings, embed_dim, commitmentCost=0.25, gradDecay=0.999, eps=1e-5):
        super().__init__()
        self.commitmentCost = commitmentCost
        self.decay = gradDecay
        self.eps = eps

        init_bound = 1 / nEmbeddings
        embedding = torch.Tensor(nEmbeddings, embed_dim)
        embedding.uniform_(-init_bound, init_bound)
        # Buffers are parameters that should not be touched by the optimiser, and these will be
        # passed back to the start of the quantisation layer, not the start of the model (i.e. the encoder).
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(nEmbeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, inp):
        nEmbed, Dim = self.embedding.size()
        inpFlat = inp.detach().reshape(-1, Dim)

        # Calculate the distance between the input and all embeddings
        embedDistances = (-torch.cdist(inpFlat, self.embedding, p=2)) ** 2

        # Fetch the closest embedding
        i = torch.argmin(embedDistances.float(), dim=-1)

        # Replace the feature map vectors with the closest embeddings.
        q = F.embedding(i, self.embedding)
        q = q.view_as(inp)
        return q, i.view(inp.size(0), inp.size(1))

    def retrieve(self, random_i):
        q = F.embedding(random_i, self.embedding)
        q = q.transpose(1, 3)
        return q

    def forward(self, inp):
        nEmbed, Dim = self.embedding.size()
        inpFlat = inp.detach().reshape(-1, Dim)

        # Calculate the distance between the input and all embeddings
        embedDistances = (-torch.cdist(inpFlat, self.embedding, p=2)) ** 2

        # Fetch the closest embedding
        i = torch.argmin(embedDistances.float(), dim=-1)

        # One hot encoding is necessary as the model is being trained over a categorial distribution
        encodings = F.one_hot(i, nEmbed).float()

        # Replace the feature map vectors with the closest embeddings.
        q = F.embedding(i, self.embedding)
        q = q.view_as(inp)

        # For passing the gradients back to the start
        if self.training:
            # Calculate the exponential moving average and apply a constant to avoid reaching a saddle point (Kang, 2024)
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
            n = torch.sum(self.ema_count)
            # Normalise the EMA count
            self.ema_count = (self.ema_count + self.eps) / (n + nEmbed * self.eps) * n

            # Update model weights
            dw = torch.matmul(encodings.t(), inpFlat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        # Calculating the difference between the feature map and the feature map after embeddings
        reconstrLoss = F.mse_loss(inp.detach(), q)

        #  Codebook loss refers to the amount of change in the interpolation distance between an embedded vector and the closest mean of cluster of embeddings, undertaken by the VQVAE during quantisation.
        codebookLoss = F.mse_loss(inp, q.detach())

        # Similar to codebook loss, however instead of moving the embedded vectors closer to the nearest cluster, 
        # the clustered embeddings are adjusted according to the embedded vector space provided by the encoder. 
        # Otherwise, the dictionary may grow in size and fail to allocate any embeddings to the feature map. 
        commitmentLoss = self.commitmentCost * reconstrLoss

        q = inp + (q - inp).detach()

        avg = torch.mean(encodings, dim=0)
        # perplexity is defined as the amount of times embeddings were accessed, 
        # not necessarily the level of entropy when referring to VQVAEs (Kang, 2024; Rosalinity, 2021)
        perplexity = torch.exp(-torch.sum(avg * torch.log(avg + 1e-10)))
        return q, commitmentLoss, codebookLoss, perplexity


"""Decodes the latent space into higher dimensionality,
   using indexed values from the dictionary of embeddings. (Yadav, 2019)"""
class Decoder(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim, kernels=(1, 3, 2, 2), stride=2):
        super().__init__()
        k1, k2, k3, k4 = kernels
        self.inner_proj = nn.Conv2d(inputDim, hiddenDim, kernel_size=1)

        self.residualConv1= nn.Conv2d(hiddenDim, hiddenDim, kernel_size=k1, padding=0)
        self.residualConv2 = nn.Conv2d(hiddenDim, hiddenDim, kernel_size=k2, padding=1)
        
        self.stridedConv1 = nn.ConvTranspose2d(hiddenDim, hiddenDim, kernel_size=k3, stride=stride, padding=0)
        self.stridedConv2 = nn.ConvTranspose2d(hiddenDim, outputDim, kernel_size=k4, stride=stride, padding=0)

    def forward(self, inp):
        inp = self.inner_proj(inp)

        out = self.residualConv1(inp)
        out += inp
        inp = nn.ReLU()(out)

        out = self.residualConv2(inp)
        out += inp
        out = nn.ReLU()(out)

        out = self.stridedConv1(out)
        out = self.stridedConv2(out)

        return out

"""
Initialise a VQVAE based on the previous classes,
process an image through the encoder, quantisation as well as decoder layers,
and output the loss values.
"""
class Model(nn.Module):
    def __init__(self, Encoder, Quantise, Decoder):
        super().__init__()
        self.encoder = Encoder
        self.quantise = Quantise
        self.decoder = Decoder

    def forward(self, inp):
        latentRep = self.encoder(inp)
        q, commitment_loss, codebook_loss, perplexity = self.quantise(latentRep)
        inpHat = self.decoder(q)
        
        return inpHat, commitment_loss, codebook_loss, perplexity
