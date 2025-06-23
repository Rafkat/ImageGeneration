import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(hidden_dim, embedding_dim, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, embedding_dim):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(hidden_dim, in_dim, kernel_size=4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, z_e):
        B, C, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        distances = (torch.sum(z_e_flat ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(z_e_flat, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices).view(B, H, W, C).permute(0, 3, 1, 2)

        commitment_loss = F.mse_loss(z_q.detach(), z_e)
        codebook_loss = F.mse_loss(z_q, z_e.detach())

        z_q = z_e + (z_q - z_e).detach()

        loss = codebook_loss + self.commitment_cost * commitment_loss
        return z_q, loss


class VQVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, embedding_dim, num_embeddings, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_dim, hidden_dim, embedding_dim)
        self.vectorQuantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(in_dim, hidden_dim, embedding_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss = self.vectorQuantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss


def vqvae_loss(x_recon, x, vq_loss, recon_weight=1.0):
    recon_loss = F.binary_cross_entropy(x_recon, x) * recon_weight
    total_loss = recon_loss + vq_loss
    return total_loss


if __name__ == '__main__':
    model = VQVAE(in_dim=3, hidden_dim=1024, embedding_dim=512, num_embeddings=10)
    model(torch.randn(2, 3, 64, 64))
