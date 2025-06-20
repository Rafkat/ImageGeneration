import torch
from torch.functional import F
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(in_dim + embedding_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=-1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        z = mu + torch.randn_like(logvar) * torch.exp(logvar * 0.5)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, embedding_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + embedding_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, in_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = torch.cat((x, y), dim=-1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class VanillaVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, embedding_dim, nb_classes):
        super(VanillaVAE, self).__init__()
        self.encoder = Encoder(in_dim, hidden_dim, latent_dim, embedding_dim)
        self.decoder = Decoder(in_dim, hidden_dim, latent_dim, embedding_dim)

        self.label_embedding = nn.Embedding(nb_classes, embedding_dim)

    def forward(self, x, y):
        y_emb = self.label_embedding(y)
        z, mu, logvar = self.encoder(x, y_emb)
        z = self.decoder(z, y_emb)
        return z, mu, logvar
