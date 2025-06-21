import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape, hidden_dim=128):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim * 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 4)
        self.leakyrelu3 = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(hidden_dim * 4, hidden_dim * 8)
        self.bn4 = nn.BatchNorm1d(hidden_dim * 8)
        self.leakyrelu4 = nn.LeakyReLU(0.2)
        self.fc5 = nn.Linear(hidden_dim * 8, int(img_shape[0] * img_shape[1] * img_shape[2]))
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.fc1(z)
        x = self.leakyrelu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leakyrelu2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leakyrelu3(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.leakyrelu4(x)
        x = self.fc5(x)
        x = self.tanh(x)
        x = x.view(x.size(0), *self.img_shape)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_shape, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(int(img_shape[0] * img_shape[1] * img_shape[2]), hidden_dim * 4)
        self.leakyrelu1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.leakyrelu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.leakyrelu1(x)
        x = self.fc2(x)
        x = self.leakyrelu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
