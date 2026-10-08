import torch
import torch.nn as nn
import torch.nn.functional as F

class VisuomotorPolicy(nn.Module):
    """
    Compact Vision-Action policy (SmolVLA-style architecture).
    Processes visual observation (overhead/wrist camera) + proprioception
    and predicts multi-step action chunks (dx, dy, dz, gripper).
    """
    def __init__(self, action_dim: int = 4, chunk_size: int = 5, state_dim: int = 7):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size

        # Lightweight CNN Visual Encoder (MobileNet/ConvNet style)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU()
        )

        # Proprioception encoder
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Fusion MLP / Action Chunking Head
        self.action_head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, chunk_size * action_dim)
        )

    def forward(self, img: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, C, H, W) normalized to [0, 1]
            state: (B, state_dim)
        Returns:
            actions: (B, chunk_size, action_dim)
        """
        b = img.shape[0]
        vis_feat = self.conv(img)
        proprio_feat = self.state_enc(state)

        fusion = torch.cat([vis_feat, proprio_feat], dim=-1)
        actions = self.action_head(fusion)
        return actions.view(b, self.chunk_size, self.action_dim)
