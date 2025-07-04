import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DDIM:
    def __init__(self, timesteps=1000):
        self.betas = self.linear_beta_scheduler(timesteps)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def linear_beta_scheduler(timesteps, beta_start=1e-4, beta_end=2e-2):
        return torch.linspace(beta_start, beta_end, timesteps)

    @staticmethod
    def forward_diffusion(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alpha_cumprod):
        noise = torch.randn_like(x_0)
        sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod[t].reshape(-1, 1, 1, 1)
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super(Block, self).__init__()
        self.time_mlp = nn.Linear(time_embedding_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        h = F.relu(self.conv1(x))
        time_emb = F.relu(self.time_mlp(t))
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        return self.conv2(h)


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_embedding_dim=256):
        super(Unet, self).__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        self.down1 = Block(in_channels, 64, time_embedding_dim)
        self.down2 = Block(64, 128, time_embedding_dim)
        self.up1 = Block(128 + 64, 64, time_embedding_dim)
        self.up2 = Block(64 + in_channels, out_channels, time_embedding_dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t.unsqueeze(-1).float())
        h1 = self.down1(x, t_emb)
        h2 = self.down2(F.max_pool2d(h1, 2), t_emb)
        h = F.interpolate(h2, scale_factor=2)
        h = self.up1(torch.cat([h, h1], dim=1), t_emb)
        h = self.up2(torch.cat([h, x], dim=1), t_emb)
        return h


class DiffusionDDIM:
    def __init__(self, timesteps=1000):
        super(DiffusionDDIM, self).__init__()
        self.timesteps = timesteps
        self.DDIM = DDIM(timesteps)
        self.Unet = Unet()

    def train(self, optimizer, epochs, train_data, device):
        for epoch in range(epochs):
            total_loss = 0
            for batch, _ in train_data:
                batch = batch.to(device)
                t = torch.randint(0, self.timesteps, (batch.size(0),)).to(device)
                x_noisy, noise = self.DDIM.forward_diffusion(batch, t, torch.sqrt(self.DDIM.alphas_cumprod),
                                                             torch.sqrt(1. - self.DDIM.alphas_cumprod))
                predicted_noise, _ = self.Unet(x_noisy, t)
                loss = F.mse_loss(predicted_noise, noise)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_loss = total_loss / len(train_data)
            print(f"INFO: Epoch {epoch + 1} | Loss: {avg_loss}")

    def sample(self, device, img_size=32, batch_size=16, num_samples=10, eta=0.0):
        with torch.no_grad():
            x = torch.randn((batch_size, 3, img_size, img_size)).to(device)
            step_seq = np.linspace(0, self.timesteps - 1, num_samples, dtype=int)
            step_seq_prev = np.append(-1, step_seq[:-1])

            for t, t_prev in zip(reversed(step_seq), reversed(step_seq_prev)):
                t_tensor = torch.full((batch_size,), t, device=device)
                predicted_noise, _ = self.Unet(x, t_tensor)
                alpha_t = self.DDIM.alphas_cumprod[t]
                alpha_prev = self.DDIM.alphas_cumprod[t_prev] if t_prev >= 0 else 1.0
                x0 = (x - predicted_noise * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

                sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
                noise = torch.randn_like(x) if t_prev >= 0 else torch.zeros_like(x)
                x = alpha_prev.sqrt() * x0 + (1 - alpha_prev - sigma ** 2).sqrt() * predicted_noise + sigma * noise

            x = x.clamp(-1, 1)
            x = x.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
            return x
