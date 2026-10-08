import os
import cv2
import numpy as np

def generate_sample_egocentric_video(output_path: str = "data/raw_videos/sample_demo.mp4", num_frames: int = 90):
    """
    Generates a realistic synthetic egocentric video demonstration of a human hand
    reaching, grasping an object on a table, lifting and placing it into a target container.
    Guarantees the test pipeline works end-to-end immediately even before phone videos are recorded.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    w, h = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))

    # Object location (mug/block on desk)
    obj_center = np.array([320, 300], dtype=np.float32)
    target_center = np.array([480, 240], dtype=np.float32)

    # Hand trajectory milestones
    # Phase 1: Reaching (0 to 30)
    # Phase 2: Grasping (30 to 45)
    # Phase 3: Moving to target (45 to 75)
    # Phase 4: Release & retract (75 to 90)

    for f in range(num_frames):
        # Canvas: desk wood/matte texture
        frame = np.full((h, w, 3), (200, 215, 225), dtype=np.uint8)

        # Draw desk surface texture
        cv2.line(frame, (0, 150), (w, 150), (160, 175, 185), 2)

        # Target zone (green marker)
        cv2.circle(frame, tuple(target_center.astype(int)), 35, (180, 230, 180), -1)
        cv2.putText(frame, "TARGET", (440, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 140, 50), 1)

        # Determine current hand and object position
        if f < 30:
            # Reaching from bottom-left towards object
            alpha = f / 30.0
            hand_pos = (1.0 - alpha) * np.array([120, 460]) + alpha * obj_center
            is_pinching = False
            cur_obj_pos = obj_center.copy()
        elif f < 45:
            # Grasping
            hand_pos = obj_center.copy()
            is_pinching = True
            cur_obj_pos = obj_center.copy()
        elif f < 75:
            # Transporting to target
            alpha = (f - 45) / 30.0
            hand_pos = (1.0 - alpha) * obj_center + alpha * target_center
            is_pinching = True
            cur_obj_pos = hand_pos.copy()
        else:
            # Release and retract
            alpha = (f - 75) / 15.0
            hand_pos = (1.0 - alpha) * target_center + alpha * np.array([550, 460])
            is_pinching = False
            cur_obj_pos = target_center.copy()

        # Draw object (Red cup/block)
        ox, oy = cur_obj_pos.astype(int)
        cv2.rectangle(frame, (ox - 22, oy - 22), (ox + 22, oy + 22), (40, 50, 210), -1)
        cv2.rectangle(frame, (ox - 22, oy - 22), (ox + 22, oy + 22), (20, 20, 130), 2)

        # Draw human arm / hand (skin-tone polygon & fingers)
        hx, hy = hand_pos.astype(int)
        skin_color = (140, 180, 235)  # BGR skin tone

        # Forearm coming in from bottom
        arm_pts = np.array([
            [hx - 30, hy + 20],
            [hx + 30, hy + 20],
            [hx + 60, h],
            [hx - 40, h]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [arm_pts], (120, 160, 215))

        # Palm
        cv2.circle(frame, (hx, hy), 28, skin_color, -1)

        # Fingers (Thumb and Index)
        pinch_offset = 6 if is_pinching else 22
        thumb_tip = (hx - pinch_offset, hy - 18)
        index_tip = (hx + pinch_offset, hy - 18)

        cv2.line(frame, (hx - 15, hy), thumb_tip, skin_color, 8)
        cv2.line(frame, (hx + 15, hy), index_tip, skin_color, 8)
        cv2.circle(frame, thumb_tip, 7, (120, 160, 220), -1)
        cv2.circle(frame, index_tip, 7, (120, 160, 220), -1)

        # Egocentric overlay timestamp
        cv2.putText(frame, f"EGOCENTRIC DEMO - Frame {f+1}/{num_frames}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

        out.write(frame)

    out.release()
    print(f"[Sample Video] Generated {num_frames} frames at {output_path}")

if __name__ == "__main__":
    generate_sample_egocentric_video()
