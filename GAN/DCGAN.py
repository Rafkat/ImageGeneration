import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, channels=1, img_size=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.init_size = img_size // 4
        self.l1 = nn.Linear(latent_dim, latent_dim * self.init_size * self.init_size)

        self.bn1 = nn.BatchNorm2d(128)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1)

        self.bn2 = nn.BatchNorm2d(latent_dim)
        self.leakyrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(latent_dim, latent_dim // 2, kernel_size=3, stride=1, padding=1)

        self.bn3 = nn.BatchNorm2d(latent_dim // 2)
        self.leakyrelu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(latent_dim // 2, channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.l1(z)
        x = x.view(x.size(0), self.latent_dim, self.init_size, self.init_size)

        x = self.bn1(x)
        x = self.upsample1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.leakyrelu2(x)
        x = self.upsample2(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = self.leakyrelu3(x)
        x = self.conv3(x)
        x = self.tanh(x)
        return x


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscriminatorBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, channels=1, hidden_dim=16, img_size=64):
        super(Discriminator, self).__init__()
        self.init_layer = nn.ModuleList([
            DiscriminatorBlock(channels, hidden_dim),
        ])
        self.layers = self.init_layer + nn.ModuleList([
            DiscriminatorBlock(hidden_dim * i, hidden_dim * 2 * i) for i in [1, 2, 4]
        ])

        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size * ds_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.adv_layer(x)
        return x
