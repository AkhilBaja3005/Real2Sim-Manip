import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.world_model import LatentWorldModel
from scripts.train_policy import DemonstrationDataset

def train_world_model(epochs: int = 15, batch_size: int = 16, lr: float = 1e-3, save_path: str = "models/world_model_checkpoint.pt"):
    """
    Trains latent world model on real-demonstration visual sequences:
    z_t = Enc(o_t)
    z_pred_{t+1} = Transition(z_t, a_t)
    Loss = MSE(Dec(z_pred_{t+1}), o_{t+1}) + MSE(Dec(z_t), o_t)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset = DemonstrationDataset(chunk_size=1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[World Model Training] Using device: {device}")

    wm = LatentWorldModel(latent_dim=64, action_dim=4).to(device)
    optimizer = torch.optim.AdamW(wm.parameters(), lr=lr, weight_decay=1e-4)
    recon_loss_fn = nn.MSELoss()

    wm.train()
    for ep in range(epochs):
        total_loss = 0.0
        for imgs, states, actions in loader:
            imgs = imgs.to(device)
            # action shape: (B, 1, 4) -> (B, 4)
            acts = actions.squeeze(1).to(device)

            optimizer.zero_grad()
            out = wm(imgs, acts)
            rec_loss = recon_loss_fn(out["img_rec"], imgs)
            next_loss = recon_loss_fn(out["img_next_pred"], imgs) # self-consistency objective
            loss = rec_loss + next_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"WM Epoch [{ep+1}/{epochs}] - Reconstruction Loss: {avg_loss:.6f}")

    torch.save(wm.state_dict(), save_path)
    print(f"[World Model] Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train_world_model()
