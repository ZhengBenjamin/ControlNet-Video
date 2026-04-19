import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.video_to_skeleton import video_to_freihand_skeleton_images


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert a video to FreiHAND-style skeleton conditioning images")
    parser.add_argument("--video-path", type=Path, required=True, help="Input video path")
    parser.add_argument("--output-dir", type=Path, default=Path("data/input"), help="Directory for output skeleton PNG files")
    parser.add_argument("--fps", type=float, default=5.0, help="Maximum sampling FPS")
    parser.add_argument("--output-size", type=int, default=512, help="Square output size in pixels")
    parser.add_argument("--line-thickness", type=int, default=2, help="Skeleton line thickness")
    parser.add_argument("--joint-radius", type=int, default=4, help="Skeleton joint radius")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum output frames. 0 means no limit")
    parser.add_argument(
        "--hand-landmarker-model-path",
        type=Path,
        default=Path("models/hand_landmarker.task"),
        help="Path to MediaPipe HandLandmarker .task model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = video_to_freihand_skeleton_images(
        video_path=args.video_path,
        output_dir=args.output_dir,
        max_fps=args.fps,
        output_size=args.output_size,
        line_thickness=args.line_thickness,
        joint_radius=args.joint_radius,
        max_frames=args.max_frames,
        hand_landmarker_model_path=args.hand_landmarker_model_path,
    )
    print(f"[video] Done. Wrote {generated} skeleton images")


if __name__ == "__main__":
    main()
