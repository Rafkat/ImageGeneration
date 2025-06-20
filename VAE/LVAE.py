import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class BottomUpEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dims):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for h_dim, z_dim in zip(hidden_dims, latent_dims):
            self.layers.append(nn.Sequential(
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, 2 * z_dim)  # Для mu и log_var
            ))
            prev_dim = z_dim

    def forward(self, x):
        stats = []  # Собираем (mu, log_var) для каждого уровня
        h = x
        for layer in self.layers:
            h = layer(h)
            mu, log_var = torch.chunk(h, 2, dim=-1)
            stats.append((mu, log_var))
            h = mu  # Передаем mu на следующий уровень
        return stats  # [(mu_1, log_var_1), ...]


class TopDownDecoder(nn.Module):
    def __init__(self, latent_dims, hidden_dims, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = latent_dims[0]
        for h_dim, z_dim in zip(hidden_dims[1:], latent_dims[1:]):
            self.layers.append(nn.Sequential(
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, 2 * z_dim)  # Для mu и log_var
            ))
            prev_dim = z_dim
        self.final_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, z_list, encoder_stats):
        h = z_list[0]
        for i, layer in enumerate(self.layers):
            mu_enc, log_var_enc = encoder_stats[i + 1]
            h = layer(h)
            mu_dec, log_var_dec = torch.chunk(h, 2, dim=-1)
            # Объединяем информацию энкодера и декодера
            mu = (mu_enc + mu_dec) / 2
            log_var = (log_var_enc + log_var_dec) / 2
            h = mu
        return self.final_layer(h), z_list


class LVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dims):
        super().__init__()
        self.bottom_up = BottomUpEncoder(input_dim, hidden_dims, latent_dims)
        self.top_down = TopDownDecoder(latent_dims, hidden_dims[::-1], input_dim)
        self.latent_dims = latent_dims

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Bottom-up pass
        encoder_stats = self.bottom_up(x)  # [(mu_1, log_var_1), ...]

        # Top-down pass с sampling
        z_list = []
        for i, (mu, log_var) in enumerate(encoder_stats):
            z = self.reparameterize(mu, log_var)
            z_list.append(z)

        x_recon, z_list = self.top_down(z_list, encoder_stats)
        return x_recon, encoder_stats, z_list


def lvae_loss(x, x_recon, encoder_stats, z_list, beta=1.0):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')

    kl_loss = 0
    for (mu, log_var), z in zip(encoder_stats, z_list):
        # KL между q(z_i | x) и p(z_i | z_{i-1})
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss += kl

    return recon_loss + beta * kl_loss
