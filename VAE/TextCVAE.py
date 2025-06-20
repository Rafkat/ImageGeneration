import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class TextEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, embedding_dim)

    def forward(self, text):
        with torch.no_grad():
            outputs = self.bert(**text)
        pooled = outputs.pooler_output
        return self.fc(pooled)


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_dim, embedding_dim)

        self.fc_mu = nn.Linear(embedding_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(embedding_dim * 2, latent_dim)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.cat([x, y], dim=-1)
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, embedding_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + embedding_dim, hidden_dim)
        self.unflatten = nn.Unflatten(1, (some_sizes))  # TODO доделать и ввести нормальные размеры
        self.upconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.upconv2 = nn.ConvTranspose2d(hidden_dim, in_dim, 4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, y):
        z = torch.cat([z, y], dim=-1)
        z = self.fc1(z)
        z = self.unflatten(z)
        z = self.upconv1(z)
        z = self.relu1(z)
        z = self.upconv2(z)
        z = self.sigmoid(z)
        return z


class TextCVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, embedding_dim):
        super(TextCVAE, self).__init__()
        self.text_encoder = TextEncoder(embedding_dim)
        self.image_encoder = Encoder(in_dim, hidden_dim, latent_dim, embedding_dim)
        self.decoder = Decoder(in_dim, hidden_dim, latent_dim, embedding_dim)

    def forward(self, text, image):
        text = self.text_encoder(text)
        z, mu, logvar = self.image_encoder(image, text)
        x_recon = self.decoder(z, text)
        return x_recon, mu, logvar
