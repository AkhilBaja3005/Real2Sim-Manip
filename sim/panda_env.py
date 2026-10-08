import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

class PandaPickPlaceEnv(gym.Env):
    """
    Franka Emika Panda manipulation environment.
    Simulates a 7-DoF arm with 2-finger parallel gripper over a desk.
    
    Observation Space:
      - 'robot_state': (7,) [eef_x, eef_y, eef_z, gripper_width, obj_x, obj_y, obj_z]
      - 'image': (128, 128, 3) RGB overhead / side camera view
      
    Action Space:
      - (4,) [dx, dy, dz, gripper_command]
        dx, dy, dz in [-0.05, 0.05] meters
        gripper in [-1 (open), 1 (close)]
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: str = "rgb_array", max_steps: int = 100):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

        # Workspace bounds [min, max]
        self.ws_x = (0.3, 0.7)
        self.ws_y = (-0.3, 0.3)
        self.ws_z = (0.05, 0.45)
        self.table_height = 0.05

        # Action: dx, dy, dz, gripper
        self.action_space = spaces.Box(
            low=np.array([-0.05, -0.05, -0.05, -1.0], dtype=np.float32),
            high=np.array([0.05, 0.05, 0.05, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation
        self.observation_space = spaces.Dict({
            "robot_state": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
            "image": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        })

        # Internal state
        self.eef_pos = np.array([0.5, 0.0, 0.3], dtype=np.float32)
        self.gripper_width = 0.08  # 0.08m = fully open, 0.0m = fully closed
        self.obj_pos = np.array([0.55, 0.1, 0.07], dtype=np.float32)
        self.target_pos = np.array([0.45, -0.15, 0.07], dtype=np.float32)
        self.is_grasped = False

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.eef_pos = np.array([0.45 + np.random.uniform(-0.02, 0.02),
                                 0.0 + np.random.uniform(-0.02, 0.02),
                                 0.30], dtype=np.float32)
        self.gripper_width = 0.08
        self.obj_pos = np.array([0.55 + np.random.uniform(-0.03, 0.03),
                                 0.10 + np.random.uniform(-0.03, 0.03),
                                 0.07], dtype=np.float32)
        self.target_pos = np.array([0.45, -0.15, 0.07], dtype=np.float32)
        self.is_grasped = False

        return self._get_obs(), {"is_success": False}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        d_pos = np.clip(action[:3], -0.05, 0.05)
        gripper_cmd = float(action[3])

        # Update EEF position
        self.eef_pos = np.clip(
            self.eef_pos + d_pos,
            [self.ws_x[0], self.ws_y[0], self.ws_z[0]],
            [self.ws_x[1], self.ws_y[1], self.ws_z[1]]
        )

        # Update Gripper
        if gripper_cmd > 0.0:  # closing
            self.gripper_width = max(0.0, self.gripper_width - 0.02)
        else:  # opening
            self.gripper_width = min(0.08, self.gripper_width + 0.02)

        # Physics & Grasp kinematics simulation
        dist_to_obj = np.linalg.norm(self.eef_pos - self.obj_pos)
        if dist_to_obj < 0.045 and self.gripper_width < 0.03:
            self.is_grasped = True

        if self.gripper_width >= 0.04:
            self.is_grasped = False

        # If grasped, object moves with end-effector
        if self.is_grasped:
            self.obj_pos = self.eef_pos.copy()
            self.obj_pos[2] = max(self.table_height + 0.02, self.obj_pos[2])

        # Success criterion: object placed near target position
        dist_to_target = np.linalg.norm(self.obj_pos[:2] - self.target_pos[:2])
        is_success = dist_to_target < 0.06 and self.obj_pos[2] < 0.10 and not self.is_grasped

        # Reward shaping
        reward = -dist_to_obj
        if self.is_grasped:
            reward += 1.0 - dist_to_target
        if is_success:
            reward += 10.0

        truncated = self.current_step >= self.max_steps
        terminated = bool(is_success)

        return self._get_obs(), reward, terminated, truncated, {"is_success": is_success}

    def _get_obs(self) -> Dict[str, Any]:
        robot_state = np.concatenate([
            self.eef_pos,
            [self.gripper_width],
            self.obj_pos
        ]).astype(np.float32)

        img = self.render()
        return {
            "robot_state": robot_state,
            "image": img
        }

    def render(self) -> np.ndarray:
        """Renders 128x128 synthetic camera image showing arm gripper, table, object, and target."""
        img = np.full((128, 128, 3), (240, 240, 240), dtype=np.uint8)  # White/gray table background
        
        # Grid lines on desk
        for i in range(0, 128, 16):
            img[i, :, :] = (220, 220, 220)
            img[:, i, :] = (220, 220, 220)

        # Helper to convert workspace (x, y) to pixel coordinates (u, v)
        def to_pixel(pos):
            u = int(((pos[1] - self.ws_y[0]) / (self.ws_y[1] - self.ws_y[0])) * 110 + 9)
            v = int(((pos[0] - self.ws_x[0]) / (self.ws_x[1] - self.ws_x[0])) * 110 + 9)
            return np.clip(u, 0, 127), np.clip(v, 0, 127)

        # Render target zone (Green disk)
        tu, tv = to_pixel(self.target_pos)
        import cv2
        cv2.circle(img, (tu, tv), 10, (180, 255, 180), -1)
        cv2.circle(img, (tu, tv), 10, (50, 180, 50), 1)

        # Render manipulable object (Red/Orange block or mug)
        ou, ov = to_pixel(self.obj_pos)
        obj_size = int(8 + (self.obj_pos[2] - 0.05) * 15)  # Scale with height
        cv2.rectangle(img, (ou - obj_size, ov - obj_size), (ou + obj_size, ov + obj_size), (40, 60, 220), -1)
        cv2.rectangle(img, (ou - obj_size, ov - obj_size), (ou + obj_size, ov + obj_size), (20, 20, 140), 1)

        # Render Franka Gripper fingers (Dark metallic blue/black)
        eu, ev = to_pixel(self.eef_pos)
        gap = int(self.gripper_width * 120)
        cv2.rectangle(img, (eu - gap - 4, ev - 5), (eu - gap, ev + 5), (50, 50, 50), -1)
        cv2.rectangle(img, (eu + gap, ev - 5), (eu + gap + 4, ev + 5), (50, 50, 50), -1)
        # Gripper mount bar
        cv2.line(img, (eu - gap, ev), (eu + gap, ev), (80, 80, 80), 3)

        return img
