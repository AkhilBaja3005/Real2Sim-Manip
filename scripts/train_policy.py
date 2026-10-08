import os
import glob
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.policy import VisuomotorPolicy
from sim.panda_env import PandaPickPlaceEnv

class DemonstrationDataset(Dataset):
    """Dataset of retargeted egocentric demonstrations."""
    def __init__(self, processed_dir: str = "data/processed", chunk_size: int = 5):
        self.chunk_size = chunk_size
        self.samples = []

        pkl_files = glob.glob(os.path.join(processed_dir, "*.pkl"))
        if not pkl_files:
            raise RuntimeError(f"No processed demonstration files found in {processed_dir}")

        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            images = data["images"]  # (T, 128, 128, 3)
            actions = data["actions"] # (T, 4)
            T = len(actions)

            # Build dummy/simulated proprioception for demonstration frames
            for t in range(T - chunk_size):
                img = images[t]
                act_chunk = actions[t : t + chunk_size]
                # Canonical state approximation [eef_x, eef_y, eef_z, gripper, obj_x, obj_y, obj_z]
                state = np.zeros(7, dtype=np.float32)
                state[:3] = [0.5, 0.0, 0.3] + np.sum(actions[:t, :3], axis=0) * 0.1
                state[3] = 0.08 if act_chunk[0, 3] < 0 else 0.0
                state[4:7] = [0.55, 0.1, 0.07]

                self.samples.append((img, state, act_chunk))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, state, act_chunk = self.samples[idx]
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 # (3, 128, 128)
        state_t = torch.from_numpy(state).float()
        act_t = torch.from_numpy(act_chunk).float()
        return img_t, state_t, act_t

def train_policy(epochs: int = 15, batch_size: int = 16, lr: float = 1e-3, save_path: str = "models/policy_checkpoint.pt"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset = DemonstrationDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[Training] Using device: {device}")

    model = VisuomotorPolicy(action_dim=4, chunk_size=5, state_dim=7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for imgs, states, target_actions in loader:
            imgs, states, target_actions = imgs.to(device), states.to(device), target_actions.to(device)
            optimizer.zero_grad()
            pred_actions = model(imgs, states)
            loss = criterion(pred_actions, target_actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Epoch [{ep+1}/{epochs}] - Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"[Training] Model successfully saved to {save_path}")

if __name__ == "__main__":
    train_policy()
