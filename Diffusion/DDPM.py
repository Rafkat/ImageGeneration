import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPM:
    def __init__(self, timesteps=1000):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.betas = self.linear_beta_scheduler(timesteps).to(device)

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
    def __init__(self, in_channels, out_channels, time_embedding_dim, up=False):
        super(Block, self).__init__()
        self.time_mlp = nn.Linear(time_embedding_dim, out_channels)
        if up:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, t):
        h = self.bn1(F.silu(self.conv1(x)))
        time_emb = F.silu(self.time_mlp(t))
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        h = self.bn2(F.silu(self.conv2(h)))
        return self.transform(h)


class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_embedding_dim=256):
        super(Unet, self).__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        self.down1 = Block(in_channels, 64, time_embedding_dim)
        self.down2 = Block(64, 128, time_embedding_dim)
        self.down3 = Block(128, 256, time_embedding_dim)
        self.up1 = Block(256 + 128, 128, time_embedding_dim, up=True)
        self.up2 = Block(128 + 64, 64, time_embedding_dim, up=True)
        self.up3 = Block(64 + in_channels, out_channels, time_embedding_dim, up=True)

    def forward(self, x, t):
        t_emb = self.time_mlp(t.unsqueeze(-1).float())
        h1 = self.down1(x, t_emb)
        h2 = self.down2(F.max_pool2d(h1, 2), t_emb)
        h3 = self.down3(h2, t_emb)
        h = F.interpolate(h3, scale_factor=2)
        h = self.up1(torch.cat([h, h2], dim=1), t_emb)
        h = F.interpolate(h, scale_factor=2)
        h = self.up2(torch.cat([h, h1], dim=1), t_emb)
        h = self.up3(torch.cat([h, x], dim=1), t_emb)
        return h


class DiffusionDDPM:
    def __init__(self, timesteps=1000):
        super(DiffusionDDPM, self).__init__()
        self.timesteps = timesteps
        self.DDPM = DDPM(timesteps)
        self.Unet = Unet()

    def train(self, optimizer, epochs, train_data, device):
        for epoch in range(epochs):
            total_loss = 0
            for batch, _ in train_data:
                batch = batch.to(device)
                t = torch.randint(0, self.timesteps, (batch.size(0),)).to(device)
                x_noisy, noise = self.DDPM.forward_diffusion(batch, t, torch.sqrt(self.DDPM.alphas_cumprod),
                                                             torch.sqrt(1. - self.DDPM.alphas_cumprod))
                predicted_noise = self.Unet(x_noisy, t)
                loss = F.mse_loss(predicted_noise, noise)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            avg_loss = total_loss / len(train_data)
            print(f"INFO: Epoch {epoch + 1} | Loss: {avg_loss}")

    def sample(self, device, img_size=32, batch_size=16):
        with torch.no_grad():
            x = torch.randn((batch_size, 1, img_size, img_size)).to(device)
            for t in reversed(range(self.timesteps)):
                t_tensor = torch.full((batch_size,), t, device=device)
                predicted_noise = self.Unet(x, t_tensor)
                alpha_t = self.DDPM.alphas[t]
                alpha_cumprod_t = self.DDPM.alphas_cumprod[t]
                beta_t = self.DDPM.betas[t]
                noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

                mu = 1 / torch.sqrt(alpha_t) * (x - beta_t / torch.sqrt(1 - alpha_cumprod_t) * predicted_noise)
                x = mu + torch.sqrt(beta_t) * noise
            x = x.clamp(-1, 1)
            x = x.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
            return x


if __name__ == '__main__':
    model = DiffusionDDPM()
    model.Unet(torch.randn((2, 1, 64, 64)), torch.randint(0, 1000, (2,)))
