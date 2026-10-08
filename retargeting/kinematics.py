import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

@dataclass
class HandPose:
    """Represents detected hand pose in camera frame."""
    wrist: np.ndarray          # (3,) [x, y, z] normalized camera coordinates
    thumb_tip: np.ndarray      # (3,)
    index_tip: np.ndarray      # (3,)
    middle_mcp: np.ndarray     # (3,)
    pinch_distance: float      # distance between thumb and index tips
    is_grasping: bool          # thresholded grasp flag
    confidence: float

@dataclass
class RobotAction:
    """End-effector Cartesian delta action for Franka Panda."""
    d_pos: np.ndarray          # (3,) [dx, dy, dz] in robot base frame (meters)
    d_rot: np.ndarray          # (3,) [d_roll, d_pitch, d_yaw] in radians
    gripper: float             # -1.0 (fully open) to 1.0 (fully closed)

class HandToRobotRetargeter:
    """
    Retargets human egocentric hand keypoint trajectories to Franka Emika Panda
    End-Effector (EEF) Cartesian delta commands.
    
    Coordinate Transformation:
    Human Egocentric Camera Frame:
      +X: Right
      +Y: Down
      +Z: Forward (away from camera / desk depth)
      
    Robot Base Frame (Franka Panda):
      +X: Forward along table
      +Y: Left along table
      +Z: Upwards
    """
    def __init__(
        self,
        pos_scale: Tuple[float, float, float] = (0.8, 0.8, 0.6),
        pinch_threshold: float = 0.065,
        smoothing_alpha: float = 0.7,
        deadband: float = 0.002
    ):
        self.pos_scale = np.array(pos_scale, dtype=np.float32)
        self.pinch_threshold = pinch_threshold
        self.smoothing_alpha = smoothing_alpha
        self.deadband = deadband
        
        self.prev_hand_pos: Optional[np.ndarray] = None
        self.smoothed_dpos = np.zeros(3, dtype=np.float32)
        self.prev_gripper_state: float = -1.0

    def reset(self):
        """Reset internal tracking state between demonstration trajectories."""
        self.prev_hand_pos = None
        self.smoothed_dpos = np.zeros(3, dtype=np.float32)
        self.prev_gripper_state = -1.0

    def retarget_frame(self, hand_pose: Optional[HandPose]) -> RobotAction:
        """
        Convert a single HandPose frame into a Franka RobotAction.
        If no hand is detected, emits a zero-delta maintain-state action.
        """
        if hand_pose is None:
            return RobotAction(
                d_pos=np.zeros(3, dtype=np.float32),
                d_rot=np.zeros(3, dtype=np.float32),
                gripper=self.prev_gripper_state
            )

        # Grasp detection via pinch distance
        is_pinching = hand_pose.pinch_distance < self.pinch_threshold
        gripper_cmd = 1.0 if is_pinching else -1.0
        self.prev_gripper_state = gripper_cmd

        # Hand position: use midpoint between wrist and index MCP/tip for stable tracking
        current_pos = hand_pose.wrist.copy()

        if self.prev_hand_pos is None:
            self.prev_hand_pos = current_pos
            return RobotAction(
                d_pos=np.zeros(3, dtype=np.float32),
                d_rot=np.zeros(3, dtype=np.float32),
                gripper=gripper_cmd
            )

        # Raw delta in camera frame: [dx_cam, dy_cam, dz_cam]
        cam_delta = current_pos - self.prev_hand_pos
        self.prev_hand_pos = current_pos

        # Mapping:
        # Robot X (Forward)  <- Cam Z (Forward / away from chest)
        # Robot Y (Left)     <- -Cam X (Negative of right)
        # Robot Z (Up)       <- -Cam Y (Negative of down)
        robot_dpos = np.array([
            cam_delta[2] * self.pos_scale[0],   # forward/backward
            -cam_delta[0] * self.pos_scale[1],  # left/right
            -cam_delta[1] * self.pos_scale[2]   # up/down
        ], dtype=np.float32)

        # Apply deadband to suppress hand tremor jitter
        mask = np.abs(robot_dpos) < self.deadband
        robot_dpos[mask] = 0.0

        # Exponential Moving Average (EMA) smoothing for stable robot control
        self.smoothed_dpos = (
            self.smoothing_alpha * self.smoothed_dpos +
            (1.0 - self.smoothing_alpha) * robot_dpos
        )

        return RobotAction(
            d_pos=self.smoothed_dpos.copy(),
            d_rot=np.zeros(3, dtype=np.float32),  # Default top-down grasp orientation
            gripper=gripper_cmd
        )
