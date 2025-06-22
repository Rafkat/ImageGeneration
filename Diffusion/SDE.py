import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class VPSDE:
    def __init__(self, beta_min=0.1, beta_max=20, T=1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def drift(self, x, t):
        return -0.5 * self.beta(t) * x

    def diffusion(self, t):
        return torch.sqrt(self.beta(t))

    def marginal_prob(self, x0, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min

        mean = torch.exp(log_mean_coeff) * x0
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std


class ScoreNet(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim))

        self.conv1 = nn.Conv2d(3, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(dim * 2, dim, 3, padding=1)
        self.conv_out = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        h1 = self.conv1(x) + t_emb.view(-1, t_emb.shape[1], 1, 1)
        h2 = self.conv2(F.silu(h1))
        h3 = self.conv3(F.silu(h2))
        return self.conv_out(h3)


class SDEDiffusion:
    def __init__(self):
        self.score_net = ScoreNet()
        self.sde = VPSDE()

    def train(self, dataloader, epochs, optimizer, device):
        self.score_net = self.score_net.to(device)

        for epoch in range(epochs):
            total_loss = 0
            for x, _ in tqdm(dataloader, total=len(dataloader)):
                x = x.to(device)
                loss = self.loss_fn(x)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_loss = total_loss / len(dataloader)
            print(f"INFO: Epoch {epoch} | Loss: {avg_loss}")

    def loss_fn(self, x0):
        t = torch.rand(x0.shape[0], device=x0.device) * self.sde.T
        mean, std = self.sde.marginal_prob(x0, t)
        noise = torch.randn_like(x0)
        xt = mean + std[:, None, None, None] * noise

        score = self.score_net(xt, t)

        loss_weights = std ** 2
        loss = torch.mean(loss_weights * (score - noise / std[:, None, None, None]) ** 2)
        return loss

    def euler_maruyama_sampler(self, shape, device, num_steps=500):
        x = torch.randn(shape, device=device)

        dt = self.sde.T / num_steps
        ts = torch.linspace(self.sde.T, 0, num_steps + 1, device=device)
        for t_prev, t_next in zip(ts[:-1], ts[1:]):
            t_tensor = torch.full((shape[0],), t_prev, device=device)

            score = self.score_net(x, t_tensor)

            drift = self.sde.drift(x, t_prev)
            diffusion = self.sde.diffusion(t_prev)
            x = x + (drift - diffusion ** 2 * score) * dt + diffusion * math.sqrt(dt) * torch.randn_like(x)

        return x

    def pc_sampler(self, shape, device, num_steps=500, snr=0.16):
        x = torch.randn(shape, device=device)

        dt = self.sde.T / num_steps
        ts = torch.linspace(self.sde.T, 0, num_steps + 1, device=device)

        for t_prev, t_next in zip(ts[:-1], ts[1:]):
            t_tensor = torch.full((shape[0],), t_prev, device=device)

            score = self.score_net(x, t_tensor)
            drift = self.sde.drift(x, t_prev)
            diffusion = self.sde.diffusion(t_prev)
            x_pred = x + (drift - diffusion ** 2 * score) * dt + diffusion * math.sqrt(dt) * torch.randn_like(x)

            grad = self.score_net(x_pred, torch.full((shape[0],), t_next, device=device))
            noise = torch.randn_like(x_pred)
            step_size = (snr * diffusion) ** 2 * 2
            x = x_pred + step_size * grad + torch.sqrt(2 * step_size) * noise

        return x
