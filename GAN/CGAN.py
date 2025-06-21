import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, nb_classes, embedding_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        self.label_embedding = nn.Embedding(nb_classes, embedding_dim)

        self.fc1 = nn.Linear(latent_dim + embedding_dim, hidden_dim)
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)

        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 4)

        self.fc4 = nn.Linear(hidden_dim * 4, int(img_shape[0] * img_shape[1] * img_shape[2]))
        self.tanh = nn.Tanh()

    def forward(self, z, labels):
        labels = self.label_embedding(labels)
        gen_input = torch.cat((z, labels), dim=-1)

        x = self.fc1(gen_input)
        x = self.leakyrelu1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.leakyrelu2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.leakyrelu3(x)
        x = self.bn3(x)
        x = self.fc4(x)
        x = self.tanh(x)
        x = x.view(-1, *self.img_shape)
        return x


class Discriminator(nn.Module):
    def __init__(self, nb_classes, hidden_dim, embedding_dim, img_shape):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(nb_classes, embedding_dim)
        input_dim = img_shape[0] * img_shape[1] * img_shape[2] + embedding_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.leakyrelu1 = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)

        self.fc3 = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels):
        x = x.view(x.size(0), -1)
        label_embed = self.label_embedding(labels)
        d_input = torch.cat((x, label_embed), dim=-1)

        x = self.fc1(d_input)
        x = self.leakyrelu1(x)
        x = self.fc2(x)
        x = self.leakyrelu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
