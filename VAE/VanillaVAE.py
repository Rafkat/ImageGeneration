import torch
from torch.functional import F
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        z = mu + torch.randn_like(logvar) * torch.exp(logvar * 0.5)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, in_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class VanillaVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim):
        super(VanillaVAE, self).__init__()
        self.encoder = Encoder(in_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(in_dim, hidden_dim, latent_dim)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        z = self.decoder(z)
        return z, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0, loss_type='bce'):
    if loss_type == 'bce':
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    else:
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * KLD
