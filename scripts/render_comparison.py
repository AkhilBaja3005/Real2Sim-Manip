import os
import cv2
import torch
import numpy as np
from sim.panda_env import PandaPickPlaceEnv
from models.policy import VisuomotorPolicy

def render_side_by_side_video(
    real_video_path: str = "data/raw_videos/sample_demo.mp4",
    policy_path: str = "models/policy_checkpoint.pt",
    output_video_path: str = "outputs/comparison_rollout.mp4",
    max_steps: int = 90
):
    """
    Renders a side-by-side comparison video:
    Left: Applicant's Egocentric Hand Manipulation Demonstration
    Right: Franka Panda Robot Arm Rollout in Simulation
    """
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # 1. Load real demo video frames
    cap = cv2.VideoCapture(real_video_path)
    real_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        real_frames.append(frame)
    cap.release()

    if not real_frames:
        raise RuntimeError(f"No frames loaded from {real_video_path}")

    # 2. Setup simulation environment
    env = PandaPickPlaceEnv(render_mode="rgb_array", max_steps=max_steps)
    obs, _ = env.reset(seed=42)

    # 3. Load trained policy
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    policy = VisuomotorPolicy(action_dim=4, chunk_size=5, state_dim=7).to(device)
    if os.path.exists(policy_path):
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        policy.eval()
        print(f"[Visualizer] Loaded policy from {policy_path}")

    # Video parameters
    target_h = 360
    target_w = 480
    combined_w = target_w * 2
    combined_h = target_h + 60 # Extra top header bar

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (combined_w, combined_h))

    sim_frames = []
    step_count = min(len(real_frames), max_steps)
    
    action_chunk_buffer = []

    for t in range(step_count):
        # Policy action prediction
        if not action_chunk_buffer:
            img_tensor = torch.from_numpy(obs["image"]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            state_tensor = torch.from_numpy(obs["robot_state"]).float().unsqueeze(0)
            with torch.no_grad():
                pred_chunks = policy(img_tensor.to(device), state_tensor.to(device)).squeeze(0).cpu().numpy()
            action_chunk_buffer = list(pred_chunks)

        action = action_chunk_buffer.pop(0) if action_chunk_buffer else np.zeros(4, dtype=np.float32)

        # Step simulation
        obs, reward, terminated, truncated, info = env.step(action)
        sim_frame_rgb = obs["image"]
        sim_frame_bgr = cv2.cvtColor(sim_frame_rgb, cv2.COLOR_RGB2BGR)

        # Scale frames to unified dimensions
        left_view = cv2.resize(real_frames[t], (target_w, target_h))
        right_view = cv2.resize(sim_frame_bgr, (target_w, target_h))

        # Compose output frame
        canvas = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)

        # Top banner
        banner = np.full((60, combined_w, 3), (35, 35, 35), dtype=np.uint8)
        cv2.putText(banner, "REAL-TO-SIM MANIPULATION: EGOCENTRIC DEMO -> FRANKA PANDA", (30, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        canvas[:60, :] = banner

        # Insert views
        canvas[60:60+target_h, :target_w] = left_view
        canvas[60:60+target_h, target_w:] = right_view

        # Subtitles / Overlays
        cv2.putText(canvas, "[REAL] Egocentric Human Demo", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(canvas, f"[SIM] Panda Robot Arm (Step {t+1})", (target_w + 20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        out.write(canvas)

    out.release()
    print(f"[Visualizer] Successfully generated side-by-side comparison video: {output_video_path}")

if __name__ == "__main__":
    render_side_by_side_video()
