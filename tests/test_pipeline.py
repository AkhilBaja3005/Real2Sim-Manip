import pytest
import numpy as np
import torch
from retargeting.kinematics import HandToRobotRetargeter, HandPose
from retargeting.hand_tracker import HandTracker
from sim.panda_env import PandaPickPlaceEnv
from models.policy import VisuomotorPolicy
from models.world_model import LatentWorldModel

def test_hand_retargeter():
    retargeter = HandToRobotRetargeter()
    pose1 = HandPose(
        wrist=np.array([0.5, 0.5, 0.5]),
        thumb_tip=np.array([0.48, 0.45, 0.5]),
        index_tip=np.array([0.52, 0.45, 0.5]),
        middle_mcp=np.array([0.5, 0.5, 0.5]),
        pinch_distance=0.04,
        is_grasping=True,
        confidence=0.9
    )
    act1 = retargeter.retarget_frame(pose1)
    assert act1.gripper == 1.0  # Pinching should trigger gripper close

    pose2 = HandPose(
        wrist=np.array([0.55, 0.48, 0.6]),
        thumb_tip=np.array([0.45, 0.45, 0.6]),
        index_tip=np.array([0.55, 0.45, 0.6]),
        middle_mcp=np.array([0.5, 0.5, 0.6]),
        pinch_distance=0.10,
        is_grasping=False,
        confidence=0.9
    )
    act2 = retargeter.retarget_frame(pose2)
    assert act2.gripper == -1.0  # Open hand should trigger gripper open
    assert len(act2.d_pos) == 3

def test_panda_env_rollout():
    env = PandaPickPlaceEnv()
    obs, info = env.reset(seed=42)
    assert "robot_state" in obs
    assert "image" in obs
    assert obs["image"].shape == (128, 128, 3)

    action = np.array([0.01, 0.0, -0.01, -1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs["robot_state"].shape == (7,)

def test_visuomotor_policy_forward():
    policy = VisuomotorPolicy(action_dim=4, chunk_size=5, state_dim=7)
    dummy_img = torch.randn(2, 3, 128, 128)
    dummy_state = torch.randn(2, 7)
    actions = policy(dummy_img, dummy_state)
    assert actions.shape == (2, 5, 4)

def test_world_model_forward():
    wm = LatentWorldModel(latent_dim=64, action_dim=4)
    dummy_img = torch.randn(2, 3, 128, 128)
    dummy_act = torch.randn(2, 4)
    out = wm(dummy_img, dummy_act)
    assert "z_t" in out
    assert "z_next_pred" in out
    assert out["img_next_pred"].shape == (2, 3, 128, 128)
