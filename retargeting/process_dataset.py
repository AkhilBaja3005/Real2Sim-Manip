import os
import cv2
import pickle
import numpy as np
from retargeting.hand_tracker import HandTracker
from retargeting.kinematics import HandToRobotRetargeter

def process_video(video_path: str, output_pkl: str, debug_video_path: str = None):
    """
    Reads egocentric video, tracks human hand keypoints and grasp states,
    and retargets them into Franka Panda Cartesian delta actions.
    Saves the processed trajectory as a dictionary with observations & actions.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    tracker = HandTracker()
    retargeter = HandToRobotRetargeter()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_writer = None
    if debug_video_path:
        os.makedirs(os.path.dirname(debug_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(debug_video_path, fourcc, fps, (w, h))

    frames_rgb = []
    actions = []
    raw_poses = []

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        hand_pose, annotated = tracker.process_frame(frame_bgr)
        robot_action = retargeter.retarget_frame(hand_pose)

        # Action: [dx, dy, dz, gripper]
        act_vec = np.array([
            robot_action.d_pos[0],
            robot_action.d_pos[1],
            robot_action.d_pos[2],
            robot_action.gripper
        ], dtype=np.float32)

        # Downsample visual frame to 128x128 for policy & world model training
        frame_rgb = cv2.cvtColor(cv2.resize(frame_bgr, (128, 128)), cv2.COLOR_BGR2RGB)

        frames_rgb.append(frame_rgb)
        actions.append(act_vec)
        raw_poses.append(hand_pose)

        if out_writer:
            # Add action overlay text to debug video
            cv2.putText(
                annotated,
                f"Act: dx={act_vec[0]:.2f} dy={act_vec[1]:.2f} dz={act_vec[2]:.2f} grip={act_vec[3]:.1f}",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )
            out_writer.write(annotated)

        frame_idx += 1

    cap.release()
    if out_writer:
        out_writer.release()

    trajectory_data = {
        "video_path": video_path,
        "num_frames": len(frames_rgb),
        "images": np.array(frames_rgb, dtype=np.uint8), # (T, 128, 128, 3)
        "actions": np.array(actions, dtype=np.float32),  # (T, 4)
    }

    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(trajectory_data, f)

    print(f"[Process Video] Successfully retargeted {len(actions)} frames from {video_path} -> {output_pkl}")
    return trajectory_data

if __name__ == "__main__":
    process_video(
        video_path="data/raw_videos/sample_demo.mp4",
        output_pkl="data/processed/sample_demo.pkl",
        debug_video_path="data/processed/annotated_sample_demo.mp4"
    )
