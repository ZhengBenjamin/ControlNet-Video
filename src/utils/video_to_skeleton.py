from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

"""
Script that converts a video of a hand into sequence of skeleton images as input to ControlNet
Uses MediaPipe HandLandmarker for hand detection + skeleton extraction, renders into image for training
"""

DEFAULT_OUTPUT_SIZE = 512
DEFAULT_MAX_FPS = 5.0
DEFAULT_HAND_LANDMARKER_MODEL_PATH = Path("models") / "hand_landmarker.task"
DEFAULT_HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_BONES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

FINGER_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 64, 64),
    (255, 160, 64),
    (255, 220, 64),
    (64, 200, 255),
    (120, 120, 255),
)


def _finger_idx(joint_idx: int) -> int:
    if 1 <= joint_idx <= 4:
        return 0
    if 5 <= joint_idx <= 8:
        return 1
    if 9 <= joint_idx <= 12:
        return 2
    if 13 <= joint_idx <= 16:
        return 3
    return 4


def _is_numbered_png(path: Path) -> bool:
    return path.suffix.lower() == ".png" and path.stem.isdigit()


def _clear_existing_numbered_pngs(output_dir: Path) -> int:
    deleted = 0
    for path in output_dir.glob("*.png"):
        if not _is_numbered_png(path):
            continue
        path.unlink()
        deleted += 1
    return deleted


def _select_best_hand(results: object) -> Optional[Sequence[object]]:
    task_landmarks = getattr(results, "hand_landmarks", None)
    if task_landmarks:
        task_handedness = getattr(results, "handedness", None)
        if not task_handedness or len(task_handedness) != len(task_landmarks):
            return task_landmarks[0]

        best_idx = 0
        best_score = -1.0
        for idx, classes in enumerate(task_handedness):
            if not classes:
                continue
            score = float(classes[0].score)
            if score > best_score:
                best_score = score
                best_idx = idx
        return task_landmarks[best_idx]

    multi_landmarks = getattr(results, "multi_hand_landmarks", None)
    if not multi_landmarks:
        return None

    multi_handedness = getattr(results, "multi_handedness", None)
    if not multi_handedness or len(multi_handedness) != len(multi_landmarks):
        return multi_landmarks[0].landmark

    best_idx = 0
    best_score = -1.0
    for idx, handedness in enumerate(multi_handedness):
        classifications = getattr(handedness, "classification", None)
        if not classifications:
            continue
        score = float(classifications[0].score)
        if score > best_score:
            best_score = score
            best_idx = idx

    return multi_landmarks[best_idx].landmark


def _landmarks_to_uv(landmarks: Sequence[object], output_size: int) -> np.ndarray:
    points = np.zeros((21, 2), dtype=np.float32)
    max_coord = float(output_size - 1)
    for idx in range(min(21, len(landmarks))):
        x = np.clip(float(landmarks[idx].x), 0.0, 1.0) * max_coord
        y = np.clip(float(landmarks[idx].y), 0.0, 1.0) * max_coord
        points[idx] = (x, y)
    return points


def _draw_skeleton(uv_points: np.ndarray, output_size: int, line_thickness: int, joint_radius: int) -> np.ndarray:
    canvas = np.zeros((output_size, output_size, 3), dtype=np.uint8)

    for j0, j1 in HAND_BONES:
        p0 = tuple(np.round(uv_points[j0]).astype(int).tolist())
        p1 = tuple(np.round(uv_points[j1]).astype(int).tolist())
        color = FINGER_COLORS[_finger_idx(j1)]
        cv2.line(canvas, p0, p1, color, thickness=line_thickness, lineType=cv2.LINE_AA)

    for idx, point in enumerate(uv_points):
        x, y = np.round(point).astype(int)
        color = (255, 255, 255) if idx == 0 else FINGER_COLORS[_finger_idx(idx)]
        cv2.circle(canvas, (x, y), joint_radius, color, thickness=-1, lineType=cv2.LINE_AA)

    return canvas


def _download_hand_landmarker_model(model_path: Path) -> None:
    """Download HandLandmarker model file to local path"""
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[video] Downloading HandLandmarker model to {model_path}")
    urllib.request.urlretrieve(DEFAULT_HAND_LANDMARKER_MODEL_URL, model_path)
    print("[video] HandLandmarker model download completed")


class HandSkeletonizer:
    """MediaPipe-based hand skeleton renderer for RGB/BGR images."""

    def __init__(
        self,
        hand_landmarker_model_path: Path = DEFAULT_HAND_LANDMARKER_MODEL_PATH,
        output_size: int = DEFAULT_OUTPUT_SIZE,
        line_thickness: int = 2,
        joint_radius: int = 4,
    ) -> None:
        self.hand_landmarker_model_path = hand_landmarker_model_path
        self.output_size = output_size
        self.line_thickness = line_thickness
        self.joint_radius = joint_radius
        self._landmarker = None

    def __enter__(self) -> "HandSkeletonizer":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def open(self) -> None:
        """Initialize HandLandmarker once for repeated inference."""
        if self._landmarker is not None:
            return

        if not self.hand_landmarker_model_path.exists():
            _download_hand_landmarker_model(self.hand_landmarker_model_path)

        try:
            from mediapipe.tasks import python as mp_tasks_python
            from mediapipe.tasks.python import vision as mp_tasks_vision
        except Exception as exc:
            raise RuntimeError("MediaPipe Tasks API is required for hand detection") from exc

        base_options = mp_tasks_python.BaseOptions(model_asset_path=str(self.hand_landmarker_model_path))
        hand_options = mp_tasks_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_tasks_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        self._landmarker = mp_tasks_vision.HandLandmarker.create_from_options(hand_options)

    def close(self) -> None:
        """Release HandLandmarker resources."""
        if self._landmarker is None:
            return

        self._landmarker.close()
        self._landmarker = None

    def render_skeleton_from_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Render hand skeleton image from BGR frame."""
        if self._landmarker is None:
            self.open()
        assert self._landmarker is not None

        frame_resized = cv2.resize(frame_bgr, (self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)

        landmarks = _select_best_hand(result)
        if landmarks is None:
            return np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)

        uv = _landmarks_to_uv(landmarks, self.output_size)
        return _draw_skeleton(
            uv_points=uv,
            output_size=self.output_size,
            line_thickness=self.line_thickness,
            joint_radius=self.joint_radius,
        )

    def render_skeleton_from_image_path(self, image_path: Path) -> np.ndarray:
        """Render hand skeleton image from file path."""
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Unable to read image: {image_path}")
        return self.render_skeleton_from_bgr(frame_bgr)


def video_to_freihand_skeleton_images(
    video_path: Path,
    output_dir: Path,
    max_fps: float = DEFAULT_MAX_FPS,
    output_size: int = DEFAULT_OUTPUT_SIZE,
    line_thickness: int = 2,
    joint_radius: int = 4,
    max_frames: int = 0,
    hand_landmarker_model_path: Path = DEFAULT_HAND_LANDMARKER_MODEL_PATH) -> int:
    """Convert a video into FreiHAND-style skeleton conditioning images."""

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    deleted = _clear_existing_numbered_pngs(output_dir)
    if deleted:
        print(f"[video] Removed {deleted} existing numbered PNG files from {output_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        source_fps = 30.0

    frame_interval = 1.0 / max_fps
    next_sample_time = 0.0

    generated = 0
    read_idx = 0
    sampled_count = 0

    print(f"[video] Reading video: {video_path}")
    print(f"[video] Source FPS: {source_fps:.2f}; target FPS: {max_fps:.2f}")

    with HandSkeletonizer(
        hand_landmarker_model_path=hand_landmarker_model_path,
        output_size=output_size,
        line_thickness=line_thickness,
        joint_radius=joint_radius,
    ) as skeletonizer:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_time = read_idx / source_fps
            read_idx += 1

            if frame_time + 1e-9 < next_sample_time:
                continue

            sampled_count += 1
            skeleton = skeletonizer.render_skeleton_from_bgr(frame_bgr)

            output_path = output_dir / f"{generated}.png"
            if not cv2.imwrite(str(output_path), skeleton):
                raise RuntimeError(f"Failed to write image: {output_path}")

            generated += 1
            if generated % 50 == 0:
                print(f"[video] Generated {generated} skeleton images")

            if max_frames > 0 and generated >= max_frames:
                break

            next_sample_time += frame_interval

    cap.release()

    print(f"[video] Sampled frames: {sampled_count}")
    print(f"[video] Generated images: {generated}")
    print(f"[video] Output directory: {output_dir}")
    return generated
