import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentWorldModel(nn.Module):
    """
    Latent World Model for Robotic Manipulation.
    Predicts next visual latent representation and environment reward
    conditioned on the current latent state and robot action:
      z_{t+1}, r_t = WorldModel(z_t, a_t)
    """
    def __init__(self, latent_dim: int = 64, action_dim: int = 4):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: image (3, 128, 128) -> latent vector z (latent_dim)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# 16x16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim)
        )

        # Transition model: z_t, a_t -> z_{t+1}
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Reward / Success predictor
        self.reward_pred = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Decoder: z -> reconstructed image (3, 128, 128)
        self.decoder_fc = nn.Linear(latent_dim, 128 * 16 * 16)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),   # 128x128
            nn.Sigmoid()
        )

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        return self.encoder(img)

    def predict_next(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.transition(torch.cat([z, action], dim=-1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        feat = self.decoder_fc(z).view(b, 128, 16, 16)
        return self.decoder_conv(feat)

    def forward(self, img: torch.Tensor, action: torch.Tensor):
        z_t = self.encode(img)
        z_next_pred = self.predict_next(z_t, action)
        img_rec = self.decode(z_t)
        img_next_pred = self.decode(z_next_pred)
        reward_pred = self.reward_pred(z_next_pred)
        return {
            "z_t": z_t,
            "z_next_pred": z_next_pred,
            "img_rec": img_rec,
            "img_next_pred": img_next_pred,
            "reward_pred": reward_pred
        }
