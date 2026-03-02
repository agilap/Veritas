"""
Layer 3 — Caption ↔ Video Consistency
OpenCV extracts one keyframe every 2 seconds → CLIP scores each frame →
returns average similarity + the most suspicious frame.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional

from layers.clip_checker import embed_text, embed_image, _cosine, THRESHOLDS

KEYFRAME_INTERVAL_SEC = 2  # sample 1 frame per N seconds
MAX_FRAMES = 30            # cap to keep inference fast (~60 sec video)


def extract_keyframes(video_path: str) -> List[Tuple[float, Image.Image]]:
    """
    Returns list of (timestamp_sec, PIL.Image) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps       = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step      = max(1, int(fps * KEYFRAME_INTERVAL_SEC))
    frames    = []
    frame_idx = 0

    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            ts  = frame_idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((ts, Image.fromarray(rgb)))
        frame_idx += 1

    cap.release()
    return frames


def check_caption_video(
    caption: str, video_path: str
) -> Tuple[float, str, str, Optional[Image.Image], float]:
    """
    Returns:
        avg_sim        — mean CLIP similarity across keyframes
        flag           — human-readable verdict
        explanation    — plain-English explanation
        worst_frame    — PIL image of the most suspicious keyframe (lowest similarity)
        worst_ts       — timestamp of the worst frame in seconds
    """
    keyframes = extract_keyframes(video_path)
    if not keyframes:
        return 0.0, "❌ Could not read video", "No frames extracted.", None, 0.0

    t_vec = embed_text(caption)

    scores: List[Tuple[float, float, Image.Image]] = []  # (sim, ts, img)
    for ts, img in keyframes:
        i_vec = embed_image(img)
        sim   = _cosine(t_vec, i_vec)
        scores.append((sim, ts, img))

    avg_sim           = float(np.mean([s[0] for s in scores]))
    worst_sim, worst_ts, worst_frame = min(scores, key=lambda x: x[0])
    avg_pct           = round(avg_sim * 100, 1)

    if avg_sim >= THRESHOLDS["match"]:
        flag = "✅ Caption consistent with video"
        exp  = (
            f"Average CLIP similarity across {len(scores)} keyframes is {avg_pct}% — "
            f"the caption generally matches the video content."
        )
    elif avg_sim >= THRESHOLDS["uncertain"]:
        flag = "⚠️ Caption partially matches video"
        exp  = (
            f"Average CLIP similarity is {avg_pct}% across {len(scores)} frames. "
            f"Some frames deviate significantly — possible context mismatch."
        )
    else:
        flag = "❌ Caption does NOT match video"
        exp  = (
            f"Average CLIP similarity is only {avg_pct}% across {len(scores)} frames. "
            f"Strong evidence that this caption does not describe the video shown. "
            f"Most suspicious moment at {worst_ts:.1f}s (similarity: {round(worst_sim*100,1)}%)."
        )

    return round(avg_sim, 4), flag, exp, worst_frame, round(worst_ts, 2)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python layers/video_checker.py <caption> <video_path>")
    else:
        caption    = sys.argv[1]
        video_path = sys.argv[2]
        avg, flag, exp, wf, wt = check_caption_video(caption, video_path)
        print(f"Avg similarity: {avg:.4f}\n{flag}\n{exp}")
        if wf:
            wf.save("worst_frame.jpg")
            print(f"Worst frame saved → worst_frame.jpg  (ts={wt}s)")
