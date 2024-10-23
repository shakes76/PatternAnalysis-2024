from math import log2
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, res_channels):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.layers(x) * self.alpha + x


class ResidualStack(nn.Module):
    def __init__(self, in_channels, res_channels, nb_layers):
        super(ResidualStack, self).__init__()
        self.stack = nn.Sequential(*[ResidualBlock(in_channels, res_channels) for _ in range(nb_layers)])

    def forward(self, x):
        return self.stack(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, res_channels, nb_res_layers, downscale_factor):
        super(Encoder, self).__init__()
        assert log2(downscale_factor) % 1 == 0, "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))
        layers = []
        c_channel, n_channel = in_channels, hidden_channels // 2
        for _ in range(downscale_steps):
            layers.append(nn.Sequential(
                nn.Conv2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, hidden_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        layers.append(ResidualStack(n_channel, res_channels, nb_res_layers))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, res_channels, nb_res_layers, upscale_factor):
        super(Decoder, self).__init__()
        assert log2(upscale_factor) % 1 == 0, "Upscale must be a power of 2"
        upscale_steps = int(log2(upscale_factor))
        layers = [nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1)]
        layers.append(ResidualStack(hidden_channels, res_channels, nb_res_layers))
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        for _ in range(upscale_steps):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, out_channels
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class VectorQuantizer(nn.Module):
    def __init__(self, in_channels, embed_dim, nb_entries):
        super(VectorQuantizer, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, embed_dim, 1)
        self.dim = embed_dim
        self.n_embed = nb_entries
        self.decay = 0.99
        self.eps = 1e-5
        embed = torch.randn(embed_dim, nb_entries, dtype=torch.float32)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(nb_entries, dtype=torch.float32))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x):
        x = self.conv_in(x).permute(0, 2, 3, 1)
        flatten = x.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()
        return quantize.permute(0, 3, 1, 2), diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class VQVAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, res_channels, nb_res_layers, embed_dim, nb_entries, downscale_factor):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels, res_channels, nb_res_layers, downscale_factor)
        self.code_layer = VectorQuantizer(hidden_channels, embed_dim, nb_entries)
        self.decoder = Decoder(embed_dim, hidden_channels, in_channels, res_channels, nb_res_layers, downscale_factor)

    def forward(self, x):
        encoded = self.encoder(x)
        quantized, diff, _ = self.code_layer(encoded)
        decoded = self.decoder(quantized)
        return decoded, diff
